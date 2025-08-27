from liquid import Template
### Template for QA part ###
### chain of thought (default)
general_cot_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
general_cot = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

### medical RAG (trival RAG)
general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
general_medrag = Template('''
Here are the relevant documents:
{{context}}
                          
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

### discuss RAG (out implementation) ####
discuss_rag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
discuss_rag = Template(''' 
Here are the relevant documents:
{{context}}
                         
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and provide your final answer in json.
''')

### define the assistant prompt ###
recruit_prompt_init = '''You are an experienced medical expert who recruits and supervise a group of experts with diverse identity and ask them to discuss and provide necessary information for solving the given medical query.'''
recruit_prompt_continue =  Template('''
Here is the medical question:
{{question}}

You can recruit {{number}} independent experts in different medical expertise. Considering the medical question and the answer choices, list the kinds of experts you will recruit to provide the necessary medical knowledge for an accurate answer.                                                           

For example, if you want to recruit three experts, you answer can be like:
                                   
1. Pediatrician - Specializes in the medical care of infants, children, and adolescents.                           
2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions.                             
3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders.                                    

Please strictly follow the above format, and do not include any explanations, summaries, or reasons outside of the numbered list. Only return the list of experts in the specified format.                                                                   
''')

### define the prompt for recruited agent ###
instruction_prompt_agents = Template(
'''
You are a {{agent_role}} who {{description}}. Your job is to collaborate with other medical experts in a team. Given a medical question, your task is not to answer it, but to share what kind of knowledge is necessary to answer the question correctly — from the perspective of your own profession and expertise. Focus on identifying the domains, facts, concepts, or reasoning approaches that would be essential for solving the question. 
'''
)
agent_reason_discuss = Template('''
Here is the medical question:

{{question}}

As a domain-specific expert, remember your role and mission: do not attempt to answer the question or infer a final conclusion. Instead, identify the essential medical knowledge, clinical background, or reasoning steps that would be necessary to answer this question accurately.
Please reflect from your area of expertise and think step-by-step.
Please keep your response concise—ideally within 7 sentences. Focus on what is most essential.
Begin your reflection below:
''')

### define the prompt for the summary agent at the start of each round ###
summary_prom_init = '''
You are a medical assistant trained to analyze and summarize expert medical opinions. 

You will be provided with a medical question, a set of domain experts responses, and optionally, a summary from the previous round.

Your task is to synthesize these opinions — along with any prior summary if available — into a clear, structured, and clinically accurate summary. 

Do not attempt to answer the question or infer a final conclusion.
'''
summary_agent_chat = Template('''
You are a medical assistant trained to analyze and summarize expert opinions across medical specialties.

Here is the medical question the experts were responding to:

{{question}}
                              
Here are reports from several medical domain experts, along with a prior summary if available:

{{assessment}}

Your task is to carefully review and synthesize the content. Please follow these steps:

1. Read and consider each expert report thoroughly.
2. Identify key medical knowledge, background concepts, or reasoning steps that are essential to answering the original medical question.
3. Synthesize these insights into a clear and unified summary.
4. Focus only on summarizing the types of knowledge needed — do not attempt to answer the question or infer a final conclusion.

Please keep your response concise—ideally within 5 sentences. Focus on what is most essential.
Begin your reflection below:
''')

### define the prompt used for add complementary
whether_agent_continue = Template('''
You are a medical domain expert collaborating with other medical experts in a team.

Here is the medical question:

{{question}}

Here is the current summary of the knowledge helpful to answer the question:

{{summary}}

Review the summary carefully. Based on your expertise, do you believe there are:
- Important omissions?
- Incorrect or vague statements?
- Areas that could be clarified or expanded?

Please think step-by-step and provide your answer.
Keep your response concise and focused—ideally within 5 sentences. Only include what is most essential.

If you identify anything worth improving, provide your input using this exact format:
yes - [Your additional complementary knowledge]
Reasoning: [Explain your thinking clearly and step-by-step]

If you believe the summary is sufficient, respond with:
no
                                  
Do not attempt to answer the question or infer a final conclusion. Only provide the formatted response as instructed.
''')

### define the prompt used for final refiner ###
refiner_prom_init = '''
You are a medical verification agent. You will be provided with a medical question and a summary report.

Your task is to directly refine and correct the summary — not evaluate it or describe how to improve it.

Make sure the summary is factually accurate and fully relevant to the question.

Do not add any new information, explanations, or reasoning.
Do not explain your edits, suggest improvements, or include any commentary.   
Do not attempt to answer the question directly.  
Return only the finalized version of the corrected summary.
'''
refiner_agent_talk = Template('''
Here is the medical question:

{{question}}
                        
Here is the summary report:
                              
{{summary}}
                              
Please directly rewrite the summary to ensure factual accuracy and full relevance to the question.
Do not add new information or explanations.  
Do not describe what should be improved — just return the corrected summary.  
Do not attempt to answer the question or infer any conclusions.

Only output the corrected summary. Do not include commentary or formatting instructions.
''')

####define prompt for using decider agent###
retrival_decide_prompt_init = '''
You are an experienced medical assistant. You will be provided with a medical query and a paragraph of retrieved information.
Your task is to analyze whether the paragraph provides enough information to reasonably support an answer to the medical query, even if the answer is not explicitly stated, as long as the conclusion would be clear to a trained medical professional.
Carefully review the entire paragraph, then reason step-by-step before arriving at your conclusion.
Strictly output your response in the following JSON format:
Dict{"step_by_step_thinking": Str (your explanation), "answer": Str ("yes" or "no")}
'''
retrival_decide_content = Template('''
Here is the medical query:
{{question}}
                            
Here is a paragraph of retrieved information:
{{information}}

Please think step-by-step and generate your output in the following JSON format: 
Dict{"step_by_step_thinking": Str (your explanation), "answer": Str ("yes" or "no")}
''')