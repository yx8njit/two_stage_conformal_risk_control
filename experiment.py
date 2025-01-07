import argparse
import math
import random
import numpy as np

def load_data(file_path):
    data = {}
    all_scores = []
    with open(file_path) as f:
        for idx, line in enumerate(f):
            line_data = line.strip().split(',') 
            query_id = line_data[0]
            doc_id = int(line_data[1])
            label = int(line_data[2])
            score = float(line_data[3])
            all_scores.append(score)
            doc_data = (doc_id, label, score)
    
            if query_id not in data.keys():
                data[query_id] = list()
            data[query_id].append(doc_data)

    # sort the list so that they arrange in the descending order of their scores
    for query_id in data.keys():
        data[query_id] = sorted(data[query_id], key=lambda x: x[2], reverse=True)

    return data, all_scores

# Exclude non-relevant queries
def exclude_ids(data):
    exclude_ids = []
    for query_id in data.keys():
        label_set = set([d[1] for d in data[query_id]])
        if len(label_set) == 1 and label_set == {0}:
        # if 1 not in set([d[1] for d in data[query_id]]):   ## alternatively, if the label is binary, we can check if 1 is in the set.
            exclude_ids.append(query_id)
    for query_id in exclude_ids:
        del data[query_id]
    return data

# Normalize score to be [0，1]
# only needed for Q&A dataset
def normalize_score(l1_data, l2_data):
    for query_id in l1_data.keys():
        l1_data[query_id] = [(d[0], d[1], math.tanh(d[2]/average_score)) for d in l1_data[query_id]]

    for query_id in l2_data.keys():
        l2_data[query_id] = [(d[0], d[1], (d[2]/100)) for d in l2_data[query_id]]
    return l1_data, l2_data

def split_query_ids(data, split_ratio=0.5):
    query_ids = list(data.keys())
    num_queries = len(query_ids)
    num_val = int (num_queries * split_ratio)
    val_idx = set(random.sample(range(num_queries), num_val))
    val_ids = [query_ids[idx] for idx in val_idx]
    test_ids = [query_ids[idx] for idx in range(num_queries) if idx not in val_idx]
    return val_ids, test_ids

def zip_data(l1_data, l2_data, ids):
    zipped_data = {}
    for id in ids:
        zipped_data[id] = (l1_data[id], l2_data[id])
    return zipped_data

def calc_l1_risk_for_query(docs_for_query, threshold, relevance_level=1):
    ground_truth_docs = get_ground_truth_above_l(docs_for_query, relevance_level)
    fetched_docs = set([doc[0] for doc in docs_for_query if doc[2] >= threshold])

    num_fetched = len(ground_truth_docs.intersection(fetched_docs))
    loss = 1 - num_fetched / (1.0 if len(ground_truth_docs) == 0 else len(ground_truth_docs))
    return (loss, fetched_docs)

def get_ground_truth_above_l(docs_for_query, relevance_level=1):
    relevant_docs = [doc[0] for doc in docs_for_query if doc[1] >= relevance_level]
    return set(relevant_docs)

def calc_retrieval_lambda(val_data, alpha, relevance_level = 1):
    pre_lambda_val = 0
    lambda_val = 0.5
    delta = abs(pre_lambda_val  - lambda_val) 
    precision = 0.00001
    M = len(val_data.keys())
    threshold = (M + 1) * alpha - 1
    # print(threshold)
    # iteration = 0
    while delta >= precision:
        total_loss = 0 
        # print(iteration)
        # print(delta)
        # iteration += 1
        # print(lambda_val)
        for query_id, (docs_for_query, _)  in val_data.items():
            total_loss += calc_l1_risk_for_query(docs_for_query, lambda_val, relevance_level)[0]
        # print(total_loss)
        if total_loss > threshold:
            lambda_val -= delta / 2
        elif total_loss < threshold:
            lambda_val += delta / 2
        else:
            break
        pre_lambda_val = lambda_val
        delta /= 2
    return lambda_val 

# calc l2 risk for already processed l2_retained_docs and corresponding l2_ground_truth
def calc_l2_risk_for_docs(l2_retained_docs, l2_ground_truth):
    denominator = sum([1.0/math.log2(i+2) for i in range(len(l2_ground_truth))])
    common_docs = set(l2_retained_docs).intersection(set(l2_ground_truth))
    if len(common_docs) == 0:
        return 1
    else:
        nominator = sum([1.0/math.log2(i+2) for i in range(len(common_docs))])
    return 1 - nominator / denominator

# calc l2 risk for specified gamma value
def calc_l2_risk_for_query(l1_retrieved_docs, l2_docs_for_query, l2_ground_truth, gamma):
    l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= gamma and doc[0] in l1_retrieved_docs])
    return calc_l2_risk_for_docs(l2_retained_docs, l2_ground_truth)

# this is the lambda_2, i.e., E_2(lambda_2, 1) <= beta
# for actual implementation, gamma = 1-gamma, therefore E_2(lambda_2, 0) <= beta
def calc_retrieval_lambda_2(val_data, beta, relevance_level = 1):
    pre_lambda_val = 0
    lambda_val = 0.5
    delta = abs(pre_lambda_val  - lambda_val) 
    precision = 0.00001
    M = len(val_data.keys())
    threshold = (M + 1) * beta - 1
    # print(threshold)
    # iteration = 0
    while delta >= precision:
        total_loss = 0 
        # print(iteration)
        # print(delta)
        # iteration += 1
        # print(lambda_val)
        for query_id, (l1_docs_for_query, l2_docs_for_query)  in val_data.items():
            l1_retrieved_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= lambda_val])
            l2_ground_truth_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= relevance_level])
            total_loss += calc_l2_risk_for_query(l1_retrieved_docs, l2_docs_for_query, l2_ground_truth_docs, 0)
        # print(total_loss)
        if total_loss > threshold:
            lambda_val -= delta / 2
        elif total_loss < threshold:
            lambda_val += delta / 2
        else:
            break
        pre_lambda_val = lambda_val
        delta /= 2
    return lambda_val        

# evenly split the validation data into val_data_1 and val_data_2
# val_data is a dict, the format is zipped_data[id] = (l1_data[id], l2_data[id])
def split_val_data(val_data):
    val_ids = list(val_data.keys())
    random.shuffle(val_ids)
    split_index = len(val_ids) // 2
    val_ids_1 = val_ids[:split_index]
    val_ids_2 = val_ids[split_index:]

    val_data_1 = {id: val_data[id] for id in val_ids_1}
    val_data_2 = {id: val_data[id] for id in val_ids_2}
    return (val_data_1, val_data_2)


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run experiments for two stage conformal risk control.")

    # Add required arguments
    parser.add_argument(
        "--l1_file_name",
        type=str,
        required=True,
        help="The file name for l1 data."
    )
    parser.add_argument(
        "--l2_file_name",
        type=str,
        required=True,
        help="The file name for l2 data."
    )
    parser.add_argument(
        "--should_normalize_score",
        action="store_true",
        default=False,
        help="Indicate whether the scores should be normalized. Default is False."
    )
    parser.add_argument(
        "--num_iteration",
        action="store_true",
        default=10,
        help="How many time we should replicate the experiment"
    )
    parser.add_argument(
        "--is_data_split",
        action="store_true",
        default=False,
        help="Indicate whether we use the data split method"
    )
    parser.add_argument(
        "--size_weight",
        action="store_true",
        default=0.5,
        help="Weight between the two stages when calculating the total prediction size, i.e., sz = w*sz_1 + (1-w)*sz_2"
    )

    # Parse the arguments
    args = parser.parse_args()
    l1_file_name = args.l1_file_name
    l2_file_name = args.l2_file_name
    should_normalize_score = args.should_normalize_score
    iteration_times = args.num_iteration
    is_data_split = args.is_data_split
    size_weight = args.size_weight

    # Load and process the data
    l1_data, _ = load_data(l1_file_name)
    l2_data, _ = load_data(l2_file_name)

    exclude_ids(l1_data)
    exclude_ids(l2_data)

    if should_normalize_score:
        l1_data, l2_data = normalize_score(l1_data, l2_data)

    level = 1
    precision = 0.0005
    num_lambda_steps = 3

    alpha = 0.3
    summary_by_beta = {}
    # for beta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    for beta in [0.1, 0.2]:
        # print(beta)
        for iteration in range(iteration_times):     
            val_ids, test_ids = split_query_ids(l1_data, 0.5)
            val_zipped_data = zip_data(l1_data, l2_data, val_ids)
            test_zipped_data = zip_data(l1_data, l2_data, test_ids)

            if is_data_split:   # for data split method, split it evenly into part 1 and part 2
                val_zipped_data_1, val_zipped_data_2 = split_val_data(val_zipped_data)
            else:               # for non-split method, the two parts are the same, both are val_zipped data
                val_zipped_data_1, val_zipped_data_2 = val_zipped_data, val_zipped_data

            max_l1_lambda_1 = calc_retrieval_lambda(val_zipped_data_1, alpha, 1)
            max_l1_lambda_2 = calc_retrieval_lambda_2(val_zipped_data_1, beta, 1)
            max_l1_lambda = min(max_l1_lambda_1, max_l1_lambda_2)
            
            lambda_grid = np.linspace(0, max_l1_lambda, num=num_lambda_steps)
            # print(lambda_grid)
            best_prediction_size = 10000000
            best_l1_size, best_l2_size = 0, 0
            best_alpha, best_beta = 0, 0
            for l1_lambda_val in lambda_grid:
                # print('{}-{}-{}'.format(beta, iteration, l1_lambda_val))
                pre_gamma_val = 0
                l2_gamma_val = 0.5
                delta = abs(pre_gamma_val  - l2_gamma_val) 
                M = len(val_zipped_data_2.keys())
                threshold = (M + 1) * beta - 1
                while delta >= precision:
                    total_loss = 0 
                    for query_id, (l1_docs_for_query, l2_docs_for_query) in val_zipped_data_2.items():
                        l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= l1_lambda_val])
                        l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= l2_gamma_val and doc[0] in l1_fetched_docs])
                        l2_ground_truth_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= level])
                        l2_risk_for_query = calc_l2_risk_for_docs(l2_retained_docs, l2_ground_truth_docs)
                        total_loss += l2_risk_for_query
                        
                    if total_loss > threshold:
                        l2_gamma_val -= delta / 2
                    elif total_loss < threshold:
                        l2_gamma_val += delta / 2
                    else:
                        break
                    pre_gamma_val = l2_gamma_val
                    delta /= 2

                ## verify control on test data
                total_l1_size, total_l2_size = 0, 0
                total_l1_loss, total_l2_loss = 0, 0
                M_test = len(test_zipped_data.keys())
                for query_id, (l1_docs_for_query, l2_docs_for_query) in test_zipped_data.items():
                    l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= l1_lambda_val])
                    l1_ground_truth = set([doc[0] for doc in l1_docs_for_query if doc[1] >= level])
                    total_l1_size += len(l1_fetched_docs)
                    total_l1_loss += 1 - len(l1_ground_truth.intersection(l1_fetched_docs)) / (1.0 if len(l1_ground_truth) == 0 else len(l1_ground_truth))
                    l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= l2_gamma_val and doc[0] in l1_fetched_docs])
                    total_l2_size += len(l2_retained_docs)
                    l2_ground_truth_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= level])
                    l2_risk_for_query = calc_l2_risk_for_docs(l2_retained_docs, l2_ground_truth_docs)
                    total_l2_loss += l2_risk_for_query
        
                cur_l1_loss = (total_l1_loss + 1)/(M_test + 1) # note to myself: should this be total_l1_loss /M_test ?
                cur_l2_loss = (total_l2_loss + 1)/(M_test + 1)
                cur_l1_size = total_l1_size / M_test
                cur_l2_size = total_l2_size / M_test
                cur_prediction_size = size_weight * cur_l1_size + (1-size_weight) * cur_l2_size
                # print('{}:{}:{}:{}:{}:{}:{}'.format(beta,l1_lambda_val, avg_l1_size+avg_l2_size, avg_l1_size, avg_l2_size, avg_l1_loss, avg_l2_loss))
                if cur_prediction_size < best_prediction_size:
                    best_prediction_size = cur_prediction_size
                    best_l1_size = cur_l1_size
                    best_l2_size = cur_l2_size
                    best_l1_loss = cur_l1_loss
                    best_l2_loss = cur_l2_loss
                    
            print('{}:{}:{}:{}:{}:{}'.format(beta, best_prediction_size, best_l1_size, best_l2_size, best_l1_loss, best_l2_loss))
            if beta not in summary_by_beta.keys():
                summary_by_beta[beta] = list()
            summary_by_beta[beta].append((best_prediction_size, best_l1_size, best_l2_size, best_l1_loss, best_l2_loss))

    for beta, results in summary_by_beta.items():
        iteration_times = len(results)
        prediction_sizes, l1_sizes, l2_sizes, l1_losses, l2_losses = [], [], [], [], []
        for tuple in results:
            prediction_sizes.append(tuple[0])
            l1_sizes.append(tuple[1])
            l2_sizes.append(tuple[2])
            l1_losses.append(tuple[3])
            l2_losses.append(tuple[4])

        # Convert lists to numpy arrays for easy computation
        prediction_sizes = np.array(prediction_sizes)
        l1_sizes = np.array(l1_sizes)
        l2_sizes = np.array(l2_sizes)
        l1_losses = np.array(l1_losses)
        l2_losses = np.array(l2_losses)            
        
        summaries = {
            "prediction_sizes": {"mean": np.mean(prediction_sizes), "std": np.std(prediction_sizes)},
            "l1_sizes": {"mean": np.mean(l1_sizes), "std": np.std(l1_sizes)},
            "l2_sizes": {"mean": np.mean(l2_sizes), "std": np.std(l2_sizes)},
            "l1_losses": {"mean": np.mean(l1_losses), "std": np.std(l1_losses)},
            "l2_losses": {"mean": np.mean(l2_losses), "std": np.std(l2_losses)},
        }
        # Print the summaries
        print(f"===beta:{beta}")
        for key, stats in summaries.items():
            print(f"{key}: mean = {stats['mean']:.4f}, std = {stats['std']:.4f}")    

if __name__ == "__main__":
    main()