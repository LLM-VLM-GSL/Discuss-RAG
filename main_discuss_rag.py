import argparse
from src.discuss_rag import Pre_dis
import os 
from datasets import load_dataset
from src.medrag import MedRAG
import os
from sentence_transformers import SentenceTransformer
import json
import matplotlib.pyplot as plt
import re
import numpy as np
def extract_choice(response_json):
    """
    Extracts only the letter choice (A, B, C, or D) from the LLM's JSON response.
    
    Args:
        response_json (str): The raw JSON response as a string.
    
    Returns:
        str: The extracted answer choice (A, B, C, or D) or 'Invalid' if not found.
    """
    try:
        response = json.loads(response_json)
        answer_choice = response.get("answer_choice", "").strip()  # Get the answer choice
        
        # Use regex to extract only the first valid letter (A, B, C, D)
        match = re.match(r"^\s*([A-D])\b", answer_choice, re.IGNORECASE)
        
        if match:
            return match.group(1).upper()  # Ensure it's uppercase A, B, C, or D
        else:
            return "Invalid"  # Return 'Invalid' if no valid choice is found
    except json.JSONDecodeError:
        return "Invalid"

def main_discuss_RAG(arg, subject = "college_medicine", test_all = False, wrong_root_dir = None, save_dir = None, save_dir_metrics = None, k_number = 4, 
         llm_model= 'OpenAI/gpt-3.5-turbo-0125', rag = False, 
         retriever_name="MedCPT", corpus_name="Textbooks", corpus_cache=True,
         rrf_k = 100, snippets=None, snippets_ids=None, question_index_ask = 0,
         HNSW=False, model_info = 'gpt-3.5', example_number = 2, use_discuss_rag = False, early_stop = None, use_verifier = False):
    '''
        arg, model_info , example_number refer to parameter in Pis_dis, and llm_model should be align with model_info
        test_all control whether test on the whole data
        interaction_txt used to save the discussion and role infor only for discussion_rag
        save_dir:folder used to save response, snippets, and score js
        save_dir_metrics: use to save subject metrics (for MMLU-Med), and total metrics
    '''
    ## define the MedRAG:
    if rag == False:
        model = MedRAG(llm_name=llm_model, rag=rag)

    elif rag == True and use_discuss_rag == False: ### the trival RAG
        model = MedRAG(llm_name=llm_model, rag=rag, retriever_name=retriever_name,
                       corpus_cache=corpus_cache, corpus_name=corpus_name,
                        HNSW=HNSW, use_discuss_rag = False)
        
    else: ### directly use discussion RAG
        model = MedRAG(llm_name=llm_model, rag=rag, retriever_name=retriever_name,
                       corpus_cache=corpus_cache, corpus_name=corpus_name,
                        HNSW=HNSW, use_discuss_rag=True
                       )
    ## define the data
    if arg.dataset == 'MMLU-Med':
        subject_base = ["anatomy", "clinical_knowledge","college_biology","college_medicine","professional_medicine","medical_genetics"]
        if subject == None: ## if no subject specific, raise value error
            raise ValueError(f'subject can not be empty')
        else: ### return either subject list or all list
            if test_all == True: ### test all samples
                
                accuracy_subject = {}
                correct_subject = {}
                answer_list =['A','B','C','D']
                base_save_dir = save_dir
                for data_category in subject: ### subject should be a list of str
                    assert data_category in subject_base
                    data_subject = load_dataset("cais/mmlu", f'{data_category}', cache_dir="/home/local/ASURITE/xdong64/Desktop/EMNLP/2025_5_19/MedRAG/MMLU_med")
                    correct_number = 0
                    total_number_subject = 0
                    if base_save_dir is not None: ## renew the save_dir based on different category
                        if rag == False:
                            save_dir = os.path.join(base_save_dir, f'{data_category}_rag_{rag}')   ## need to renew the save_dir every time visit the new subject since the dict changed
                            os.makedirs(save_dir,exist_ok=True)
                        else:
                            save_dir = os.path.join(base_save_dir,f'{data_category}_rag_{rag}_K{k_number}')
                            os.makedirs(save_dir,exist_ok=True)
                    
                    # arg.interaction_txt = os.path.join(save_dir_metrics,f'{data_category}_rag_{rag}_K{k_number}.txt')
                    args.interation_txt = os.path.join(save_dir,f"{data_category}_rag_{rag}_K{k_number}.txt")

                    for i in range(len(data_subject['test'])):
                        response_path = os.path.join(save_dir, f'response_{i}.json')
                        if os.path.exists(response_path):
                            print(f'question {i} already exists')
                            continue
                        total_number_subject += 1
                        #####
                        question_i = data_subject['test'][i]['question'] ##str
                        choice_i = data_subject['test'][i]['choices'] ##list
                        answer_index_i = answer_list[data_subject['test'][i]['answer']] ## int - correct answer
                        ## transfer choice into dict that fulfill requirements
                        choices_dict_i = {chr(65 + i): f"{choice}" for i, choice in enumerate(choice_i)}
                        if rag == False:
                            predict_answer_i, _, _ = model.answer(question=question_i, options=choices_dict_i, save_dir=save_dir, question_index_ask = i) ## dict
                            predict_answer_i = extract_choice(predict_answer_i) ### extract the meaningful answer choice from raw json
                            if answer_index_i == predict_answer_i.strip():
                                correct_number +=1
                        
                        elif rag == True and use_discuss_rag == False:
                            predict_answer_i, _, _ = model.answer(question=question_i, options=choices_dict_i, k = k_number, save_dir = save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask = i) ## dict
                            predict_answer_i = extract_choice(predict_answer_i) ## return should be like A. xxx.
                            if answer_index_i == predict_answer_i.strip():
                                correct_number +=1
                        else:
                            ### first define the  discuss agent object
                            discuss_agent = Pre_dis(question_i, example_number, model_info, arg)
                            suggestions, _, _, _ = discuss_agent.process_info_query(question_index_ask = i)
                            predict_answer_i, _, _ = model.answer(background=suggestions, question=question_i, options=choices_dict_i, k = k_number, save_dir = save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask = i, use_verifier = use_verifier) ## dict
                            predict_answer_i = extract_choice(predict_answer_i) ## return should be like A. xxx.
                            if answer_index_i == predict_answer_i.strip():
                                correct_number +=1

                    accuracy_subject_i =  correct_number / len(data_subject['test'])
                    accuracy_subject[data_category] = accuracy_subject_i
                    correct_subject[data_category] = correct_number
                    if save_dir_metrics:
                        metrics_name = f'metrics_discuss_rag_{rag}.txt' if rag==False else f'metrics_discuss_rag_{rag}_k_{k_number}.txt'
                        with open(os.path.join(save_dir_metrics, metrics_name), 'a') as f:
                            f.write(f'\n---------->\n')
                            f.write(f'Accuracy for:{data_category}:{accuracy_subject_i}\n')
                            f.write(f'Correct number for:{data_category}:{correct_number}\n')
                            f.write(f'Total subect number:{total_number_subject}')
                            f.write(f'\n---------->\n')

                if save_dir_metrics:
                    metrics_path = os.path.join(save_dir_metrics)
                    os.makedirs(metrics_path, exist_ok=True)
                    metrics_name = f'metrics_discuss_rag_{rag}.txt' if rag==False else f'metrics_discuss_rag_{rag}_k_{k_number}.txt'
                    # Save metrics
                    with open(os.path.join(metrics_path, metrics_name), 'a') as f:
                        f.write(f'accuracy: {accuracy_subject}\n')
                        f.write(f'correct_number: {correct_subject}\n')
            else:
                assert wrong_root_dir is not None
                ### load all test subject, check all subject element in wrong_root_dir
                correct_subject = {}
                answer_list =['A','B','C','D']
                base_save_dir = save_dir
                for data_category in subject:
                    if not os.path.exists(os.path.join(wrong_root_dir, data_category)):
                        raise ValueError(f'{wrong_root_dir} does not contain wrong answer for {data_category}')
                    ### collect the index of wrong list
                    wrong_answer_list = []
                    for file in os.listdir(os.path.join(wrong_root_dir, data_category)):
                        if file.startswith('response_'):
                            wrong_answer_list.append(int(file.split('.')[0].split('_')[1]))
                    #reorder to wrong memeber index
                    wrong_answer_list = sorted(wrong_answer_list)
                    ### begin main part to do discuss rag
                    assert data_category in subject_base
                    data_subject = load_dataset("cais/mmlu", f'{data_category}', cache_dir="/home/local/ASURITE/xdong64/Desktop/EMNLP/2025_5_19/MedRAG/MMLU_med")
                    correct_number = 0
                    if base_save_dir is not None: ## renew the save_dir based on different category
                        if rag == False:
                            save_dir = os.path.join(base_save_dir, f'{data_category}_rag_{rag}')   ## need to renew the save_dir every time visit the new subject since the dict changed
                        else:
                            save_dir = os.path.join(base_save_dir,f'{data_category}_rag_{rag}_K{k_number}')

                    args.interation_txt = os.path.join(save_dir,f"{data_category}_rag_{rag}_K{k_number}.txt")
                    for i in wrong_answer_list:
                        response_path = os.path.join(save_dir, f'response_{i}.json')
                        if os.path.exists(response_path):
                            print(f'question {i} already exists')
                            continue
                        question_i = data_subject['test'][i]['question'] ##str
                        choice_i = data_subject['test'][i]['choices'] ##list
                        answer_index_i = answer_list[data_subject['test'][i]['answer']] ## int - correct answer
                        ## transfer choice into dict that fulfill requirements
                        choices_dict_i = {chr(65 + i): f"{choice}" for i, choice in enumerate(choice_i)}
                        if rag == False:
                            predict_answer_i, _, _ = model.answer(question=question_i, options=choices_dict_i, save_dir=save_dir, question_index_ask = i) ## dict
                            predict_answer_i = extract_choice(predict_answer_i) ### extract the meaningful answer choice from raw json
                            if answer_index_i == predict_answer_i.strip():
                                correct_number +=1
                        else:
                            ### first define the  discuss agent object
                            discuss_agent = Pre_dis(question_i, example_number, model_info, arg)
                            suggestions, _, _, _ = discuss_agent.process_info_query(question_index_ask = i)

                            predict_answer_i, _, _ = model.answer(background=suggestions, question=question_i, options=choice_dict_all, k = k_number, save_dir = save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask = i) ## dict
                            predict_answer_i = extract_choice(predict_answer_i) ## return should be like A. xxx.
                            if answer_i == predict_answer_i.strip():
                                correct_number +=1


                    correct_subject[data_category] = correct_number
                    if save_dir_metrics:
                        metrics_name = f'metrics_discuss_rag_{rag}.txt' if rag==False else f'metrics_discuss_rag_{rag}_k_{k_number}.txt'
                        with open(os.path.join(save_dir_metrics, metrics_name), 'a') as f:
                            f.write(f'\n---------->\n')
                            f.write(f'Correct number for:{data_category}:{correct_number}\n')
                            f.write(f'\n---------->\n')
                
                if save_dir_metrics:
                    metrics_path = os.path.join(save_dir_metrics)
                    os.makedirs(metrics_path, exist_ok=True)
                    metrics_name = f'metrics_discuss_rag_{rag}.txt' if rag==False else f'metrics_discuss_rag_{rag}_k_{k_number}.txt'
                    # Save metrics
                    with open(os.path.join(metrics_path, metrics_name), 'a') as f:
                        f.write(f'correct_number: {correct_subject}\n')

    elif arg.dataset == 'BioASQ':
        answer_list = ["yes", "no"]
        compare_list = ["A", "B"]
        arg.interaction_txt = os.path.join(save_dir,f'{arg.dataset}_rag_{rag}_K{k_number}.txt')

        with open(arg.data_path,'r') as f:
            calculated_number = 0
            dataset = json.load(f) ## should be a list
            correct_number = 0
            ### modify for test wrong QA
            if test_all == False and wrong_root_dir is not None:
                wrong_qa_index = []
                for file in os.listdir(wrong_root_dir):
                    if file.startswith('response_'):
                        wrong_qa_index.append(int(file.split(".")[0].split("_")[1]))
                filtered_dataset = [dataset[i] for i in wrong_qa_index]
                original_indices = wrong_qa_index

            else:
                filtered_dataset = dataset
                original_indices = list(range(len(dataset)))
            ###
            # total_number = len(filtered_dataset) ### the total number of question
            total_number = 0
            ## only yes, no choice for all question ###
            choice_dict_all = {chr(65 + i): f"{choice}" for i, choice in enumerate(answer_list)}
            for new_i, question_i_dict in enumerate(filtered_dataset):
                i = original_indices[new_i]
                response_path = os.path.join(save_dir, f'response_{i}.json')
                if early_stop is not None and calculated_number > early_stop:
                    break
                if os.path.exists(response_path):
                    print(f'question {i} already exists')
                    continue
                ### if the response is not exist, increate total_number ###
                total_number +=1
                ## get the question and answer
                question_i = question_i_dict['question']
                answer_i = compare_list[question_i_dict['answer']]
                ### if no rag is used
                if rag == False:
                    calculated_number +=1
                    predict_answer_i, _, _ = model.answer(question=question_i, options=choice_dict_all, save_dir=save_dir, question_index_ask = i) ## dict
                    predict_answer_i = extract_choice(predict_answer_i) ### extract the meaningful answer choice from raw json
                    if answer_i == predict_answer_i.strip():
                        correct_number +=1
                ### just use trival reg ###
                elif rag == True and use_discuss_rag == False:
                    calculated_number +=1 
                    predict_answer_i, _, _ = model.answer(question=question_i, options=choice_dict_all, k = k_number, save_dir = save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask = i) ## dict
                    predict_answer_i = extract_choice(predict_answer_i) ## return should be like A. xxx.
                    if answer_i == predict_answer_i.strip():
                        correct_number +=1
                ### just use discuss reg ###
                elif rag == True and use_discuss_rag == True:
                    calculated_number +=1
                    discuss_agent = Pre_dis(question_i, example_number, model_info, arg)
                    suggestions, _, _, _ = discuss_agent.process_info_query(question_index_ask = i)

                    predict_answer_i, _, _ = model.answer(background=suggestions, question=question_i, options=choice_dict_all, k = k_number, save_dir = save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask = i) ## dict
                    predict_answer_i = extract_choice(predict_answer_i) ## return should be like A. xxx.
                    if answer_i == predict_answer_i.strip():
                        correct_number +=1

            if save_dir_metrics:
                metrics_name = f'metrics_rag_{rag}.txt' if rag==False else f'metrics_discuss_{use_discuss_rag}_rag_{rag}_k_{k_number}.txt'
                with open(os.path.join(save_dir_metrics, metrics_name), 'a') as f:
                    f.write(f'\n---------->\n')
                    f.write(f"RAG:{rag} / Discussion_RAG: {use_discuss_rag}")
                    f.write(f'Correct number for:{correct_number} / {total_number}\n')
                    f.write(f"Accuracy is:{correct_number/total_number}")
                    f.write(f'\n---------->\n')

    elif arg.dataset == 'PubMedQA':
        answer_list = ["yes", "no", "maybe"]
        compare_list = ["A", "B", "C"]
        arg.interaction_txt = os.path.join(save_dir, f'{arg.dataset}_rag_{rag}_K{k_number}.txt')

        with open(arg.data_path, 'r') as f:
            calculated_number = 0
            dataset = json.load(f)  # should be a list
            correct_number = 0

            # Modify for test wrong QA
            if test_all == False and wrong_root_dir is not None:
                wrong_qa_index = []
                for file in os.listdir(wrong_root_dir):
                    if file.startswith('response_'):
                        wrong_qa_index.append(int(file.split(".")[0].split("_")[1]))
                filtered_dataset = [dataset[i] for i in wrong_qa_index]
                original_indices = wrong_qa_index
            else:
                filtered_dataset = dataset
                original_indices = list(range(len(dataset)))

            total_number = 0  # will be counted based on responses actually attempted
            choice_dict_all = {chr(65 + i): f"{choice}" for i, choice in enumerate(answer_list)}

            for new_i, question_i_dict in enumerate(filtered_dataset):
                i = original_indices[new_i]
                response_path = os.path.join(save_dir, f'response_{i}.json')

                if early_stop is not None and calculated_number > early_stop:
                    break
                if os.path.exists(response_path):
                    print(f'question {i} already exists')
                    continue

                total_number += 1
                question_i = question_i_dict['question']
                answer_i = compare_list[question_i_dict['answer']]

                if rag == False:
                    calculated_number += 1
                    predict_answer_i, _, _ = model.answer(question=question_i, options=choice_dict_all, save_dir=save_dir, question_index_ask=i)
                    predict_answer_i = extract_choice(predict_answer_i)
                    if answer_i == predict_answer_i.strip():
                        correct_number += 1

                elif rag == True and use_discuss_rag == False:
                    calculated_number += 1
                    predict_answer_i, _, _ = model.answer(question=question_i, options=choice_dict_all, k=k_number, save_dir=save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask=i)
                    predict_answer_i = extract_choice(predict_answer_i)
                    if answer_i == predict_answer_i.strip():
                        correct_number += 1

                elif rag == True and use_discuss_rag == True:
                    calculated_number +=1
                    discuss_agent = Pre_dis(question_i, example_number, model_info, arg)
                    suggestions, _, _, _ = discuss_agent.process_info_query(question_index_ask = i)

                    predict_answer_i, _, _ = model.answer(background=suggestions, question=question_i, options=choice_dict_all, k = k_number, save_dir = save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask = i) ## dict
                    predict_answer_i = extract_choice(predict_answer_i) ## return should be like A. xxx.
                    if answer_i == predict_answer_i.strip():
                        correct_number +=1

            if save_dir_metrics:
                metrics_name = f'metrics_rag_{rag}.txt' if rag == False else f'metrics_discuss_{use_discuss_rag}_rag_{rag}_k_{k_number}.txt'
                with open(os.path.join(save_dir_metrics, metrics_name), 'a') as f:
                    f.write(f'\n---------->\n')
                    f.write(f"RAG:{rag} / Discussion_RAG: {use_discuss_rag}\n")
                    f.write(f'Correct number for:{correct_number} / {total_number}\n')
                    f.write(f"Accuracy is:{correct_number/total_number}")
                    f.write(f'\n---------->\n')
    
    elif arg.dataset == 'MedQA_US':
        arg.interaction_txt = os.path.join(save_dir, f'{arg.dataset}_rag_{rag}_K{k_number}.txt')
        with open(arg.data_path, 'r') as f:
            calculated_number = 0
            dataset = [json.loads(line) for line in f]  # should be a list of dict
            correct_number = 0

            # Modify for test wrong QA
            if test_all == False and wrong_root_dir is not None:
                wrong_qa_index = []
                for file in os.listdir(wrong_root_dir):
                    if file.startswith('response_'):
                        wrong_qa_index.append(int(file.split(".")[0].split("_")[1]))
                filtered_dataset = [dataset[i] for i in wrong_qa_index] ## fitered list of dict
                original_indices = wrong_qa_index
            else:
                filtered_dataset = dataset
                original_indices = list(range(len(dataset)))

            total_number = 0  # will be counted based on responses actually attempted
            
            for new_i, question_i_dict in enumerate(filtered_dataset):
                i = original_indices[new_i]
                response_path = os.path.join(save_dir, f'response_{i}.json')

                if early_stop is not None and calculated_number > early_stop:
                    break
                if os.path.exists(response_path):
                    print(f'question {i} already exists')
                    continue

                total_number += 1
                question_i = question_i_dict['question'] # str
                choice_i = question_i_dict['options'] ### shown options
                answer_i = question_i_dict['answer_idx'] #str of answer, e.g. D

                if rag == False:
                    calculated_number += 1
                    predict_answer_i, _, _ = model.answer(question=question_i, options=choice_i, save_dir=save_dir, question_index_ask=i)
                    predict_answer_i = extract_choice(predict_answer_i)
                    if answer_i == predict_answer_i.strip():
                        correct_number += 1

                elif rag == True and use_discuss_rag == False:
                    calculated_number += 1
                    predict_answer_i, _, _ = model.answer(question=question_i, options=choice_i, k=k_number, save_dir=save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask=i)
                    predict_answer_i = extract_choice(predict_answer_i)
                    if answer_i == predict_answer_i.strip():
                        correct_number += 1

                elif rag == True and use_discuss_rag == True:
                    calculated_number +=1
                    discuss_agent = Pre_dis(question_i, example_number, model_info, arg)
                    suggestions, _, _, _ = discuss_agent.process_info_query(question_index_ask = i)

                    predict_answer_i, _, _ = model.answer(background=suggestions, question=question_i, options=choice_i, k = k_number, save_dir = save_dir, snippets=snippets, snippets_ids=snippets_ids, question_index_ask = i) ## dict
                    predict_answer_i = extract_choice(predict_answer_i) ## return should be like A. xxx.
                    if answer_i == predict_answer_i.strip():
                        correct_number +=1
            
            if save_dir_metrics:
                metrics_name = f'metrics_rag_{rag}.txt' if rag == False else f'metrics_discuss_{use_discuss_rag}_rag_{rag}_k_{k_number}.txt'
                with open(os.path.join(save_dir_metrics, metrics_name), 'a') as f:
                    f.write(f'\n---------->\n')
                    f.write(f"RAG:{rag} / Discussion_RAG: {use_discuss_rag}\n")
                    f.write(f'Correct number for:{correct_number} / {total_number}\n')
                    f.write(f"Accuracy is:{correct_number/total_number}")
                    f.write(f'\n---------->\n')


def calculate_accuracy_folder_Pub_Bio(dataset, data_path, root_path, save_path, rag, discussion_rag=False, k_number=None, stop_number = 5000):
    correct_number = 0 
    total_question = 0 
    answer_list = ["A","B"] if dataset == "BioASQ" else ["A", "B", "C"]
    with open(data_path,'r') as g:
        question = json.load(g) ## list
        for i, question_dict in enumerate(question):
            target_path = os.path.join(root_path, f'response_{i}.json')
            correct_answer = answer_list[question_dict["answer"]] ###A or B for BioASQ or A,B,C for PubMedQA
            if not os.path.exists(target_path):
                continue
            else:
                total_question +=1
                with open(target_path,'r') as f:
                    response_i = json.load(f)[0]
                    predict_answer = extract_choice(response_i)
                    if correct_answer == predict_answer:
                        correct_number +=1
            if total_question > stop_number:
                break
    metrics_name = f'metrics_rag_{rag}.txt' if rag==False else f'metrics_discuss_{discussion_rag}_rag_{rag}_k_{k_number}.txt'
    with open(os.path.join(save_path, metrics_name), 'a') as f:
        f.write(f'\n---------->\n')
        f.write(f"RAG:{rag} / Discussion_RAG: {discussion_rag}")
        f.write(f'Correct number for:{correct_number} / {total_question}\n')
        f.write(f"Accuracy is:{correct_number/total_question}")
        f.write(f'\n---------->\n')

def calculate_accuracy_folder_MedQA(dataset, data_path, root_path, save_path, rag, discussion_rag=False, k_number=None, stop_number = 5000):
    correct_number = 0 
    total_question = 0 
    answer_list = ["A","B","C","D"]
    with open(data_path,'r') as g:
        question = [json.loads(line) for line in g]

    for index, question_dict in enumerate(question):
        target_path = os.path.join(root_path, f'response_{index}.json')
        answer_i = question_dict['answer_idx']
        if not os.path.exists(target_path):
            print(f" response for {index} does not exists ")
            continue
        else:
            total_question +=1
            with open(target_path,'r') as f:
                response_i = json.load(f)[0]
                predict_answer = extract_choice(response_i)
                if  answer_i == predict_answer:
                        correct_number +=1
            if total_question > stop_number:
                break
    metrics_name = f'metrics_rag_{rag}.txt' if rag==False else f'metrics_discuss_{discussion_rag}_rag_{rag}_k_{k_number}.txt'
    with open(os.path.join(save_path, metrics_name), 'a') as f:
        f.write(f'\n---------->\n')
        f.write(f"RAG:{rag} / Discussion_RAG: {discussion_rag}")
        f.write(f'Correct number for:{correct_number} / {total_question}\n')
        f.write(f"Accuracy is:{correct_number/total_question}")
        f.write(f'\n---------->\n')


 

def calculate_accuracy_folder_MMLU(subject_list, root_path, save_path, rag, k_number):
    subject_base = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "professional_medicine", "medical_genetics"]
    accuracy_subject = {}
    correct_subject = {}
    answer_list = ['A', 'B', 'C', 'D']

    total_correct_number = 0
    total_question = 0

    for subject in subject_list:
        assert subject in subject_base, f"{subject} is not in supported subjects"

        data_subject = load_dataset("cais/mmlu", subject, cache_dir="/home/local/ASURITE/xdong64/Desktop/EMNLP/2025_5_19/MedRAG/MMLU_med")
        test_data = data_subject['test']
        total_question += len(test_data)
        if rag == True:
            subject_path = os.path.join(root_path, f'{subject}_rag_{rag}_K{k_number}')
        else:
            subject_path = os.path.join(root_path, f'{subject}_rag_{rag}')

        if not os.path.exists(subject_path):
            raise ValueError(f'{subject_path} does not exist')

        correct_number = 0
        for i, sample in enumerate(test_data):
            gt_answer = answer_list[sample['answer']]
            response_path = os.path.join(subject_path, f'response_{i}.json')

            if not os.path.exists(response_path):
                raise ValueError(f'{response_path} is missing')
            
            with open(response_path, 'r') as f:
                response_json = json.load(f)
                response_json = response_json[0]
                predicted_answer = extract_choice(response_json)
                if predicted_answer == gt_answer:
                    correct_number += 1

        accuracy = correct_number / len(test_data)
        correct_subject[subject] = correct_number
        accuracy_subject[subject] = round(accuracy, 4)
        total_correct_number += correct_number

        print(f'Correct number and accuracy of {subject}: {correct_number} / {accuracy:.4f}')

    with open(save_path, 'a') as f:
        f.write(f'subject accuracy: {accuracy_subject}\n')
        f.write(f'subject correct_number: {correct_subject}\n')
        f.write(f'total question: {total_question}\n')
        f.write(f'total accuracy: {total_correct_number / total_question:.4f}\n')
        f.write(f'total correct number: {total_correct_number}\n')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Parse argument in discuss")
    parser.add_argument('--dataset', type = str, default='MMLU-Med', help = ' the exampler and test dataset, can only be MMLU-Med, BioASQ, PubMedQA and MedQA_US')
    parser.add_argument('--data_path', type = str, default='', help = ' the data path to json file, can only be used in BioASQ and PubMedQA')
    parser.add_argument('--num_rounds', type = int, default=2, help = 'number of round of discussion')
    parser.add_argument('--num_turns', type = int, default=2, help = 'number of turns of discussion per round')
    parser.add_argument('--interaction_txt', type = str, default = '', help = 'the path to where store the whole conversation')
    parser.add_argument('--number', type = int, default=3, help = 'number of agent used')
    args = parser.parse_args()

    subject = ["anatomy", "clinical_knowledge","college_biology","college_medicine","professional_medicine","medical_genetics"]
    rag = True
    save_dir = "./MMLU_savedir"
    save_dir_metrics = "./MMLU_savedir_metric"
    main_discuss_RAG(args, subject=subject, test_all=True, wrong_root_dir='', save_dir = save_dir, save_dir_metrics = save_dir_metrics, k_number=9, rag = rag, use_discuss_rag = True,use_verifier =True )


    root_path = './MMLU_result'
    save_path = './metrics_MMLU.txt'
    calculate_accuracy_folder_MMLU(subject, root_path, save_path, True, 9)