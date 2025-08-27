import openai
import os
import json
import random
from src.config import config
# from src.medrag import openai_client_with_retries
from pptree import Node
# from template import * ### need to modify the prompt template
from src.template_discuss import *
from datasets import load_dataset
from prettytable import PrettyTable 
import re
import time

def openai_client_with_retries(**kwargs):
    max_retries = 5
    wait_time = 5  # Start with 5 seconds

    for attempt in range(max_retries):
        try: ### directly return the response content ###
            response = openai.ChatCompletion.create(**kwargs)
            return response["choices"][0]["message"]["content"]
        except openai.error.Timeout:
            print(f"Timeout occurred. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(60)  # Wait for 1 minute before retrying
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return None
    return None


### define the openai environment ###
openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]

###define the basic agent class ###
class Agent:
    '''
        Here, we define the basic agent class to define how to make conversation
    '''
    def __init__(self, instruction, role, examples=None, model_info='gpt-3.5'):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.client = openai_client_with_retries  ### the return actually is a pure string

        self.messages = [{"role": "system", "content": instruction}]
        if examples:
            for example in examples: ### provide some history conversation ###
                self.messages.append({"role": "user", "content": example['question']})
                self.messages.append({"role": "assistant", "content": example['answer'] + "\n\n" + example['reason']})

    def chat(self, message, chat_mode=True):
        model_name = 'gpt-3.5-turbo-0125' if self.model_info == 'gpt-3.5' else 'gpt-4o-mini'
        self.messages.append({"role": "user", "content": message})

        try:
            response = self.client(
                model=model_name,
                messages=self.messages,
                request_timeout=30
            )
            # print(response)
            self.messages.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return None

    def temp_responses(self, message):
        model_name = 'gpt-3.5-turbo-0125' if self.model_info == 'gpt-3.5' else 'gpt-4o-mini'
        self.messages.append({"role": "user", "content": message})

        temperatures = [0.0]
        responses = {}
        for temperature in temperatures:
            try:
                response = self.client(
                    model=model_name,
                    messages=self.messages,
                    temperature=temperature,
                    request_timeout=30
                )
                responses[temperature] = response
            except Exception as e:
                print(f"Error during OpenAI API call at temperature {temperature}: {e}")
                responses[temperature] = None

        return responses[0.0]
    

    
### define the main discuss rag part ###
class Pre_dis:
    def __init__(self, question, examplers, model, args):
        self.question = question
        self.examplers = examplers ### some questions, here since we directly use MMLU, so we set this as number of example
        self.model_info = model ### LLM model we use / try 'gpt-4o-mini'
        self.args = args

    def is_valid_format(self, agent_str): 
        return bool(re.match(r'^\d+\.\s+.+?\s*[-–]\s+.+', agent_str))
    
    def process_info_query(self, question_index_ask=None):
        print("[INFO] Step 1. Expert Recruitment")
        recruit_prom_init = recruit_prompt_init ##control number of assistant agent
        recruit_prom_cont = recruit_prompt_continue.render(question=self.question, number = self.args.number).strip()
        #### the leader agent, ensure a valid recruitment following format
        tmp_agent = Agent(instruction=recruit_prom_init, role = 'Hiring Lead', model_info = self.model_info )
        tmp_agent.chat(recruit_prom_init) 
                
        max_retry = 7
        valid_recruitment = False
        # Save a clean copy of messages
        initial_messages = tmp_agent.messages.copy()
        for attempt in range(max_retry):
            # Restore clean state before retrying
            tmp_agent.messages = initial_messages.copy()

            recruited = tmp_agent.chat(recruit_prom_cont)

            agents_info = [agent_info for agent_info in recruited.split('\n') if agent_info]

            if all(self.is_valid_format(agent) for agent in agents_info) and len(agents_info) == self.args.number:
                valid_recruitment = True
                break
            else:
                print(f"[Attempt {attempt+1}/{max_retry}] Invalid format. Retrying...")
        if not valid_recruitment:
            print(f"Recruitment failed after max retries for question {question_index_ask}. You may need to manually mentor the result.")
            raise ValueError(f'invalid recruitment')
            
        ###define symbol for agent identity ###
        agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
        random.shuffle(agent_emoji)
        ###define the identity for recruited agent ###
        agent_list = ""
        for i, agent in enumerate(agents_info):
            agent_role = agent.split('-')[0].split('.')[1].strip().lower() ## collect role string, i.e.,Pediatrician
            description = agent.split('-')[1].strip().lower()
            agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
        ### then create agent with these roles
        agent_dict = {}
        medical_agents = []
        for agent in agents_info:
            try:
                agent_role = agent.split('-')[0].split('.')[1].strip().lower() ## collect role string, i.e.,Pediatrician
                description = agent.split('-')[1].strip().lower()
            except:
                continue
            instruction_prom_agent = instruction_prompt_agents.render(agent_role = agent_role, description = description).strip()
            inst_agent = Agent(instruction=instruction_prom_agent, role = agent_role, model_info=self.model_info)
            inst_agent.chat(instruction_prom_agent)
            ### collect member agent initialization
            agent_dict[agent_role] = inst_agent
            medical_agents.append(inst_agent)
        ### print the recruit information
        for j, agent in enumerate(agents_info):
            try:
                print(f"Agent {j+1} ({agent_emoji[j]} {agent.split('-')[0].strip()}): {agent.split('-')[1].strip()}")
            except:
                print(f"Agent {j+1} ({agent_emoji[j]}): {agent}")

        ### place left for few-shot exampler ###

        ###
        ### stage two sequentially add discussion ###
        print()
        print("[INFO] Step 2. Collaborative Decision Making for Necessary Knowledge")
        print()

        num_rounds = self.args.num_rounds ## how many round totally##
        ### basically how exists 1 turn per rounds ###
        num_turns = self.args.num_turns  ## how many allowable discussion turn for each round ##
        num_agents = len(medical_agents)
        interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}
        
        print()
        print("[INFO] Step 2.1. Participatory Debate")
        print()
        ## since we got intiail opinion, start from 1 and add 1 more round opinion
        ### since +3 to track the last possible iteration
        round_opinions = {n: {} for n in range(1, num_rounds+3)}
        # round_answers = {f"Round {n}": None for n in range(1, num_rounds+3)}
        initial_report = ""
        
        ### collect initial opinion ###
        for k, v in agent_dict.items(): ##k: agent_role, v: agent_object
            opinion = v.chat(agent_reason_discuss.render(question = self.question).strip())
            initial_report += f"({k.lower()}): {opinion}\n"
            round_opinions[1][k.lower()] = opinion ### update the opinion from each hired agent
        ### create the medical_assistant key to track summary, and new summary only based on renew one
        for n in range(1, num_rounds+2):
            round_opinions[n]['medical_assistant'] = None
            ##round_opinion[i] {{agent_role}: summary} / round_1 store the intialized summary
        
        final_summary = ""

        for n in range(1, num_rounds+2):
            print(f"== Round {n} ==")
            # round_name = f"Round {n}"
            ### start from each round, summarize agent response first ###

            ## round opinion track agent response for each round ##
            ## hire a summary agent to summary others opinion##
            summary_prompt = summary_prom_init.strip()
            agent_summary = Agent(instruction = summary_prompt, role = 'medical_assistant', model_info=self.model_info)
            agent_summary.chat(summary_prompt)
            ###assessment work to collect each round report from each agent and use for summary agent to summary ###
            ###escape the summary agent, new summary only based on possible new content and previous content###
            assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items() if k !="medical_assistant")
            ### now summary should also add previous medical_assistant summary report ###
            if n != 1:
                assessment += f"(medical_assistant): {round_opinions[n-1]['medical_assistant']}\n"

            report = agent_summary.chat(summary_agent_chat.render(assessment=assessment, question = self.question).strip())
            ### renew summary result current round summary ###
            round_opinions[n]['medical_assistant'] = report
            ## track summary report
            final_summary = report
            num_yes = 0
            ###now hired agent should provide whether they want to add complement ###
            ### then sequentailly iterate agent list to track whether they want to add comments ###
            for idx, v in enumerate(medical_agents): ##v: class Agent
                ### based on round report sequentially add comments if they want ###
                whether_continue = whether_agent_continue.render(question = self.question, summary = report).strip()
                participate = v.chat(whether_continue)
                print(f'{v.role.lower()} : {participate}')
                if "yes" in participate.lower().strip():
                    num_yes += 1
                    ### modify the round opinion for this hired agent ###
                    round_opinions[n+1][f'{v.role.lower()}'] = participate

                else: ### don't want to add complement
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")
                    round_opinions[n+1][f'{v.role.lower()}'] = ""
                ## this turn no additional complementart ##
            if num_yes == 0:
                break
        # # ### make a summary again if there still got complement add to final round
        # if any(value != "" for value in round_opinions[num_rounds+2].values()):
        #     assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[num_rounds+2].items() if k !="medical_assistant" and v != "")
        #     assessment += 
        print()
        print(f'Final summary:{final_summary}')
        print()
        ### now after discussion, we need to do final
        print()
        print("[INFO] Step 3. Final Verification for Suggestions")
        print()

        refiner = Agent(instruction = refiner_prom_init.strip(), role = "Refiner", model_info=self.model_info)
        refiner.chat(refiner_prom_init.strip())
        
        suggestions = refiner.temp_responses(refiner_agent_talk.render(question = self.question, summary = final_summary).strip())
        role_information = list(agent_dict.keys())

        ### store the generation results only if provide not none path ###
        if self.args.interaction_txt != "":
            with open(self.args.interaction_txt,'a') as w:
                w.write(f'-------->\n')
                w.write(f'Response for question:{question_index_ask}:\n')
                w.write(f'<---------->\n')
                w.write(f'Final Suggestions: {suggestions}\n')
                w.write(f'summary before final output: {final_summary}\n')
                w.write(f'Role hired: {role_information}\n')
                w.write(f'Discussion process:{round_opinions}\n')
                w.write(f'-------->\n')
                w.write(f'-------->\n')

        

        return suggestions, final_summary, role_information, round_opinions
