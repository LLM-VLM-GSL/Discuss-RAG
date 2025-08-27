import openai
import os
import json
import random
from .config import config
from .medrag import openai_client_with_retries
from pptree import Node
from template import *
from datasets import load_dataset
from prettytable import PrettyTable 
import re


openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]
### 
### here our implementation only work on gpt
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

        return responses
    
### define the pre_rag_discussion class before
class Pre_dis:
    def __init__(self, question, examplers, model, args):
        self.question = question
        self.examplers = examplers ### some questions, here since we directly use MMLU, so we set this as number of example
        self.model_info = model ### LLM model we use
        self.args = args ##(number of agent, num_rounds, num_turns)

    def is_valid_format(self, agent_str): ## define helpful function to ensure that our recruitment is valid
        return bool(re.match(r'^\d+\.\s+.+?\s*-\s+.+', agent_str))

    def process_info_query(self, question_index_ask=None):
        print("[INFO] Step 1. Expert Recruitment")
        ### define some necessary prompt
        recruit_prom_init = recruit_prompt_init ##control number of assistant agent
        recruit_prom_cont = recruit_prompt_continue.render(question=self.question, number = self.args.number).strip()
        #### the leader agent
        tmp_agent = Agent(instruction=recruit_prom_init, role = 'Hiring Lead', model_info = self.model_info )
        tmp_agent.chat(recruit_prom_init) ### hire a hiring lead agent, give him mission but not track its return

        # recruited = tmp_agent.chat(recruit_prom_cont) ###recruted number of agent and chat with recruited result
        # agents_info = [agent_info for agent_info in recruited.split('\n') if agent_info]
        # ### just in case other things besides recruit information included
        # agents_info = agents_info[:self.args.number]
        ### ensure valid recruitment with max_retry times try ###
        max_retry = 4
        valid_recruitment = False
        # Save a clean copy of messages
        initial_messages = tmp_agent.messages.copy()
        for attempt in range(max_retry):
            # Restore clean state before retrying
            tmp_agent.messages = initial_messages.copy()

            recruited = tmp_agent.chat(recruit_prom_cont)

            agents_info = [agent_info for agent_info in recruited.split('\n') if agent_info]
            # agents_info = agents_info[:self.args.number] ### we only track first number info

            if all(self.is_valid_format(agent) for agent in agents_info) and len(agents_info) == self.args.number:
                valid_recruitment = True
                break
            else:
                print(f"[Attempt {attempt+1}/{max_retry}] Invalid format. Retrying...")
        if not valid_recruitment:
            print(f"Recruitment failed after max retries for question {question_index_ask}. You may need to manually mentor the result.")
            raise ValueError(f'invalid recruitment')
        

        agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
        random.shuffle(agent_emoji)
        ## collect agent information  
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
            medical_agents.append(inst_agent) ### list of agent element
        ###check the recrutied information
        # with open('/home/local/ASURITE/xdong64/Desktop/EMNLP/2025_5_19/MedRAG/Experiments/wrong_answer_analysis/college_medicine_wrong.txt','a') as f:
        #     for k, v in agent_dict.items():
        #         f.write(f'{k},{v.messages}\n')
        ### 

        for j, agent in enumerate(agents_info):
            try:
                print(f"Agent {j+1} ({agent_emoji[j]} {agent.split('-')[0].strip()}): {agent.split('-')[1].strip()}")
            except:
                print(f"Agent {j+1} ({agent_emoji[j]}): {agent}")
        ### provide few-shot example for leader agent
        fewshot_examplers = ""
        if self.args.dataset == 'MMLU-Med' and self.examplers is not None: ###maybe need to change the template for examplers or if it's necessary?
            subject_base = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "professional_medicine", "medical_genetics"]
            selected_number = 0
            answer_list = ["A","B","C","D"]
            while selected_number < self.examplers:  ### why continue talk with "Hiring lead" agent ? what about discuss based on keywords / key sentence, that used to make summary 
                selected_subject = random.choice(subject_base) ##'str'
                data_subject = load_dataset("cais/mmlu", f'{selected_subject}', cache_dir="/home/local/ASURITE/xdong64/Desktop/EMNLP/2025_5_19/MedRAG/MMLU_med")
                question_index = random.choice(range(len(data_subject['test'])))
                selected_question = f"[Example {selected_number+1}]\n"+ data_subject['test'][question_index]['question']
                options = {chr(65 + i): f"{choice}" for i, choice in enumerate(data_subject['test'][question_index]['choices'])}
                options = [f"({k}) {v}" for k, v in options.items()]
                random.shuffle(options)
                selected_question += " " + " ".join(options) ## <question> <options>
                ## <answer> 
                selected_answer = f"({answer_list[data_subject['test'][question_index]['answer']]}) {data_subject['test'][question_index]['choices'][data_subject['test'][question_index]['answer']]}"
                few_show_example = fewshot_example.render(question=selected_question, answer=selected_answer).strip()## since template already contain Answer, just erase it
                few_shot_reason = tmp_agent.chat(f"{few_show_example}") ### this part provide necessary reasons

                ## then collect all material
                selected_question += f"\n{selected_answer}\n{few_shot_reason}\n\n" ##<answer> <reason>
                fewshot_examplers += selected_question
                selected_number += 1

        elif self.examplers is not None:
            assert self.args.dataset in ["BioASQ", "PubMedQA"]
            selected_number = 0
            answer_list = ["A" , "B"] if self.args.dataset == "BioASQ" else ["A", "B", "C"]
            option_choice = ["yes","no"] if self.args.dataset == "BioASQ" else ["yes", "no", "maybe"]
            options_list = {chr(65 + i): f"{choice}" for i, choice in enumerate(option_choice)}
            options = [f"({k}) {v}" for k, v in options_list.items()]
            random.shuffle(options)
            with open(self.args.data_path,'r') as f:
                question_list = json.load(f) ## list
                while selected_number < self.examplers:
                    question_index = random.choice(range(len(question_list))) #### change the name ###
                    selected_question = f"[Example {selected_number+1}]\n"+ question_list[question_index]["question"]
                    selected_question += " " + " ".join(options)
                    selected_answer = f"({answer_list[question_list[question_index]['answer']]}) {option_choice[question_list[question_index]['answer']]}"
                    few_show_example = fewshot_example.render(question=selected_question, answer=selected_answer).strip()
                    few_shot_reason = tmp_agent.chat(f"{few_show_example}") 
                    selected_question += f"\n{selected_answer}\n{few_shot_reason}\n\n"
                    fewshot_examplers += selected_question
                    selected_number += 1

        print()
        print("[INFO] Step 2. Collaborative Decision Making for Necessary Knowledge")
        print()

        num_rounds = self.args.num_rounds ## total number of rounds
        num_turns = self.args.num_turns ## total number of interaction per round
        num_agents = len(medical_agents) ### medical_agents is list of agent object
        ## create the whole dialogue structure
        interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

        print()
        print("[INFO] Step 2.1. Participatory Debate")
        print()
        ### let medical agent discuss and record results ###
        round_opinions = {n: {} for n in range(1, num_rounds+1)}
        round_answers = {f"Round {n}": None for n in range(1, num_rounds+1)}
        initial_report = ""

        ## collect initial discussion
        for k, v in agent_dict.items(): ##k: agent_role, v: agent_object
            opinion = v.chat(agent_reason_discuss.render(fewshot_examplers = fewshot_examplers, question = self.question).strip())
            initial_report += f"({k.lower()}): {opinion}\n"
            round_opinions[1][k.lower()] = opinion
        # with open('/home/local/ASURITE/xdong64/Desktop/EMNLP/2025_5_19/MedRAG/Experiments/wrong_answer_analysis/college_medicine_wrong.txt','a') as f:
        #     f.write(f'-------->\n')
        #     f.write(f'initialized discussion:\n')
        #     f.write(f'{initial_report}')
        final_opinion = None

        for n in range(1, num_rounds+1):
            print(f"== Round {n} ==")
            round_name = f"Round {n}"
            agent_summary = Agent(instruction = summary_prom_init, role = 'medical_assistant', model_info=self.model_info)
            agent_summary.chat(summary_prom_init)

            ### then collect the summary response from multiply agent
            assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())
            ### ask summary agent to summary the key background knowledge needed
            report = agent_summary.chat(summary_agent_chat.render(assessment=assessment).strip())

            ### further discuss among agent
            for turn_num in range(num_turns):
                turn_name = f"Turn {turn_num + 1}"
                print(f"|_{turn_name}")

                num_yes = 0
                for idx, v in enumerate(medical_agents):
                    ## track all message send to Agent {idx +1} in turn_num, num_round
                    all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())

                    participate = v.chat(whether_agent_continue.render(opinion = assessment if n ==1 else all_comments).strip())

                    if 'yes' in participate.lower().strip(): 
                        chosen_expert = v.chat(continue_agent_talk.render(agent_list=agent_list).strip())
                        chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                        for ce in chosen_experts:
                            specific_question = v.chat(convey_agent_talk.render(index = ce, role = medical_agents[ce-1].role).strip())
                            
                            ### print and log the conversation between agent###
                            print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                            interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                        num_yes +=1
                    else: ### indicate the agent that don't want to further discuss
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")
                ### now agent want to talk further, make a agreement
                print(f'{num_yes}')
                if num_yes == 0:
                    break
            ### no need for other round of discussion  
            if num_yes == 0:
                break
            tmp_final_answer  = {} ###track the conversation result if want to discuss
            for i, agent in enumerate(medical_agents):
                response = agent.chat(final_agent_opinion.render(question = self.question).strip())
                tmp_final_answer[agent.role] = response

            round_answers[round_name] = tmp_final_answer
            final_opinion = tmp_final_answer
            # with open('/home/local/ASURITE/xdong64/Desktop/EMNLP/2025_5_19/MedRAG/Experiments/wrong_answer_analysis/college_medicine_wrong.txt','a') as f:
            #     f.write(f'\n-------->\n')
            #     f.write(f'final discussion report:\n')
            #     for k, v in tmp_final_answer.items():
            #         f.write(f'Role:{k}\n')
            #         f.write(f'{v}\n')
            #         f.write(f'<--------->\n')
        with open(self.args.interaction_txt,'a') as f:
            f.write(f'\n-------->\n')
            f.write(f'round answer for:{round_answers}:\n')
            f.write(f'\n-------->\n')
        

        #### not print the interaction
        print('\nInteraction Log')        
        myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

        for i in range(1, len(medical_agents)+1):
            row = [f"Agent {i} ({agent_emoji[i-1]})"]
            for j in range(1, len(medical_agents)+1):
                if i == j:
                    row.append('')
                else:
                    i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                            for k in range(1, len(interaction_log)+1)
                            for l in range(1, len(interaction_log['Round 1'])+1))
                    j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                            for k in range(1, len(interaction_log)+1)
                            for l in range(1, len(interaction_log['Round 1'])+1))
                    
                    if not i2j and not j2i:
                        row.append(' ')
                    elif i2j and not j2i:
                        row.append(f'\u270B ({i}->{j})')
                    elif j2i and not i2j:
                        row.append(f'\u270B ({i}<-{j})')
                    elif i2j and j2i:
                        row.append(f'\u270B ({i}<->{j})')

            myTable.add_row(row)
            if i != len(medical_agents):
                myTable.add_row(['' for _ in range(len(medical_agents)+1)])
        
        print(myTable)
        with open(self.args.interaction_txt,'a') as f:
            f.write(f'\n-------->\n')
            f.write(f'Conversation Table for:{question_index_ask}:\n')
            f.write(f"{myTable}\n")
            f.write(f'\n-------->\n')


        print()
        print("[INFO] Step 3. Final Decision for Suggestions")
        print()

        moderator = Agent(instruction = decision_prom_init, role = "Moderator", model_info=self.model_info)
        moderator.chat(decision_prom_init)

        _suggestions = moderator.temp_responses(decision_agent_talk.render(question = self.question, suggestions = final_opinion).strip())
        role_information = list(agent_dict.keys()) ### list of agent hired
        with open(self.args.interaction_txt,'a') as f:
            f.write(f'\n-------->\n')
            f.write(f'Role and Suggestions for:{question_index_ask}:\n')
            f.write(f'Role selected:{role_information}:\n')
            f.write(f"{_suggestions}\n") ### return is a key
            f.write(f'\n-------->\n')
        ### modify for debug
        with open(self.args.interaction_txt,'a') as f:
            f.write(f'\n-------->\n')
            f.write(f'sugg and role for:{question_index_ask}:\n')
            f.write(f'{_suggestions[0.0]}\n')
            f.write(f'{role_information}\n')
            f.write(f'\n-------->\n')

        return _suggestions[0.0], role_information