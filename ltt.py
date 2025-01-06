from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

def hb_p_value(r_hat, n, alpha):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)
    
    def h1(y, mu, eps=1e-10):
        y = np.clip(y, eps, 1 - eps)
        mu = np.clip(mu, eps, 1 - eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))
    
    hoeffding_p_value = np.exp(-n * h1(min(r_hat, alpha), alpha))
    return min(bentkus_p_value, hoeffding_p_value)


def bonferroni(p_values,delta):
    rejections, _, _, _ = multipletests(p_values,delta,method='holm',is_sorted=False,returnsorted=False)
    R = np.nonzero(rejections)[0]
    return R    

def calc_p_values(risk_vals, n, alpha_val):
    p_values = np.zeros_like(risk_vals)
    for i in range(len(risk_vals)): 
        p_values[i] = hb_p_value(risk_vals[i],n,alpha_val)
        # print(risk_vals[i])
    return p_values


def evaluate_ltt(test_data, lambda_val, gamma_val):
    ## verify control on test data
    total_l1_size, total_l2_size = 0, 0
    total_l1_loss, total_l2_loss = 0, 0
    M_test = len(test_data.keys())
    for query_id, (l1_docs_for_query, l2_docs_for_query) in test_data.items():
        l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= lambda_val])
        l1_ground_truth = set([doc[0] for doc in l1_docs_for_query if doc[1] >= level])
        total_l1_size += len(l1_fetched_docs)
        total_l1_loss += 1 - len(l1_ground_truth.intersection(l1_fetched_docs)) / (1.0 if len(l1_ground_truth) == 0 else len(l1_ground_truth))
        l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= gamma_val and doc[0] in l1_fetched_docs])
        total_l2_size += len(l2_retained_docs)
        l2_ground_truth_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= level])
        l2_risk_for_query = calc_l2_risk_for_query(l2_retained_docs, l2_ground_truth_docs)
        total_l2_loss += l2_risk_for_query

    avg_l1_loss = (total_l1_loss + 1)/(M_test + 1)
    avg_l2_loss = (total_l2_loss + 1)/(M_test + 1)
    avg_l1_size = total_l1_size / M_test
    avg_l2_size = total_l2_size / M_test
    prediction_size = avg_l1_size + avg_l2_size

    return (avg_l1_loss, avg_l2_loss, avg_l1_size, avg_l2_size, prediction_size)

def calc_ltt_lambda_gamma(l1_risk_vals_flatten, l2_risk_vals_flatten, M, alpha, beta, num_points_on_grid):
    l1_p_values=calc_p_values(l1_risk_vals_flatten, M, alpha)
    l2_p_values=calc_p_values(l2_risk_vals_flatten, M, beta)
    R1 = bonferroni(l1_p_values, delta)
    R2 = bonferroni(l2_p_values, delta)
    gamma_indices = list()
    lambda_indices = list()
    lambda_vals = np.linspace(0, 1, num_points_on_grid)
    gamma_vals = np.linspace(0, 1, num_points_on_grid)
    for index in list(set(R1) & set(R2)):
        gamma_index= int(index / num_points_on_grid)
        gamma_indices.append(gamma_index)
        lambda_index = index % num_points_on_grid
        lambda_indices.append(lambda_index)
    max_lambda_index = max(lambda_indices)
    max_gamma_index = max(gamma_indices)
    return(lambda_vals[max_lambda_index], gamma_vals[max_gamma_index])



evaluate_ltt(test_zipped_data, 0.05, 0.13)
num_points_on_grid =100
num_iterations = 15

# Create the range for lambda and gamma
lambda_range = np.linspace(0, 1, num_points_on_grid)
gamma_range = np.linspace(0, 1, num_points_on_grid)

# Create the 2D grid
lambda_grid, gamma_grid = np.meshgrid(lambda_range, gamma_range)
l1_risk_vals = np.zeros((num_points_on_grid, num_points_on_grid))
l2_risk_vals = np.zeros((num_points_on_grid, num_points_on_grid))
l1_sizes = np.zeros((num_points_on_grid, num_points_on_grid))
l2_sizes = np.zeros((num_points_on_grid, num_points_on_grid))
total_sizes = np.zeros((num_points_on_grid, num_points_on_grid))

for k in range(num_iterations):
    val_ids, test_ids = split_query_ids(l1_data, 0.5)
    val_zipped_data = zip_data(l1_data, l2_data, val_ids)
    test_zipped_data = zip_data(l1_data, l2_data, test_ids)
    for i in range(num_points_on_grid):
        # print('{}-{}'.format(k, i))
        for j in range(num_points_on_grid):
            lambda_val = lambda_grid[i, j]
            gamma_val = gamma_grid[i, j]
            (ltt_l1_risk_val,ltt_l2_risk_val, ltt_l1_sz, ltt_l2_sz, ltt_total_sz) = evaluate_ltt(val_zipped_data, lambda_val, gamma_val)
            l1_risk_vals[i, j] += ltt_l1_risk_val
            l2_risk_vals[i, j] += ltt_l2_risk_val
            l1_sizes[i, j] += ltt_l1_sz
            l2_sizes[i, j] += ltt_l2_sz
            total_sizes[i, j] +=  ltt_total_sz
l1_risk_vals /= num_iterations
l2_risk_vals /= num_iterations
l1_sizes /= num_iterations
l2_sizes /= num_iterations
total_sizes /=  num_iterations

l1_risk_vals_flatten= l1_risk_vals.flatten()
l2_risk_vals_flatten= l2_risk_vals.flatten()

## run some test first
alpha = 0.1
beta = 0.1
l1_p_values=calc_p_values(l1_risk_vals_flatten, M, alpha)
l2_p_values=calc_p_values(l2_risk_vals_flatten, M, beta)
R1 = bonferroni(l1_p_values, delta)
R2 = bonferroni(l2_p_values, delta)
print(R1)
print(R2)

lambda_vals = np.linspace(0, 1, num_points_on_grid)
gamma_vals = np.linspace(0, 1, num_points_on_grid)
gamma_indices = list()
lambda_indices = list()
for index in list(set(R1) & set(R2)):
    gamma_index= int(index / num_points_on_grid)
    gamma_indices.append(gamma_index)
    lambda_index = index % num_points_on_grid
    lambda_indices.append(lambda_index)
    # print('{}:{}'.format(lambda_index,gamma_index))
    # print('{}:{}'.format(lambda_vals[lambda_index],gamma_vals[gamma_index]))
max_lambda_index = max(lambda_indices)
max_gamma_index = max(gamma_indices)
print(lambda_vals[max_lambda_index])
print(gamma_vals[max_gamma_index])
evaluate_ltt(test_zipped_data, lambda_vals[max_lambda_index], gamma_vals[max_gamma_index])
# print(verify_l1(test_data, lambda_vals[max_lambda_index]))
# print(verify_l2(test_data, gamma_vals[min_gamma_index], lambda_vals[max_lambda_index]))
evaluate_ltt(test_zipped_data, lambda_vals[max_lambda_index], gamma_vals[max_gamma_index])



M = len(test_zipped_data.keys())
alpha = 0.3 ## Fix alpha
num_iterations=30
for beta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    ltt_l1_risk_val_sum,ltt_l2_risk_val_sum, ltt_l1_sz_sum, ltt_l2_sz_sum, ltt_total_sz_sum=0,0,0,0,0
    for k in range(num_iterations):
        (lambda_val, gamma_val)=calc_ltt_lambda_gamma(l1_risk_vals_flatten, l2_risk_vals_flatten, M, alpha, beta, num_points_on_grid)
        
        val_ids, test_ids = split_query_ids(l1_data, 0.5)
        val_zipped_data = zip_data(l1_data, l2_data, val_ids)
        test_zipped_data = zip_data(l1_data, l2_data, test_ids)
        (ltt_l1_risk_val,ltt_l2_risk_val, ltt_l1_sz, ltt_l2_sz, ltt_total_sz) = evaluate_ltt(test_zipped_data, lambda_val, gamma_val)
        ltt_l1_risk_val_sum +=ltt_l1_risk_val
        ltt_l2_risk_val_sum+= ltt_l2_risk_val
        ltt_l1_sz_sum+=ltt_l1_sz
        ltt_l2_sz_sum+=ltt_l2_sz
        ltt_total_sz_sum+=ltt_total_sz
    ltt_l1_risk_val_avg = ltt_l1_risk_val_sum/num_iterations
    ltt_l2_risk_val_avg = ltt_l2_risk_val_sum/num_iterations
    ltt_l1_sz_avg= ltt_l1_sz_sum/num_iterations
    ltt_l2_sz_avg= ltt_l2_sz_sum/num_iterations
    ltt_total_sz_avg = ltt_total_sz_sum/num_iterations
    print('{},{},{},{},{},{}'.format(beta, ltt_l1_risk_val_avg, ltt_l2_risk_val_avg, ltt_l1_sz_avg, ltt_l2_sz_avg, ltt_total_sz_avg))


M = len(test_zipped_data.keys())
beta = 0.2 ## Fix alpha
num_iterations=30
for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    ltt_l1_risk_val_sum,ltt_l2_risk_val_sum, ltt_l1_sz_sum, ltt_l2_sz_sum, ltt_total_sz_sum=0,0,0,0,0
    for k in range(num_iterations):
        (lambda_val, gamma_val)=calc_ltt_lambda_gamma(l1_risk_vals_flatten, l2_risk_vals_flatten, M, alpha, beta, num_points_on_grid)
        
        val_ids, test_ids = split_query_ids(l1_data, 0.5)
        val_zipped_data = zip_data(l1_data, l2_data, val_ids)
        test_zipped_data = zip_data(l1_data, l2_data, test_ids)
        (ltt_l1_risk_val,ltt_l2_risk_val, ltt_l1_sz, ltt_l2_sz, ltt_total_sz) = evaluate_ltt(test_zipped_data, lambda_val, gamma_val)
        ltt_l1_risk_val_sum +=ltt_l1_risk_val
        ltt_l2_risk_val_sum+= ltt_l2_risk_val
        ltt_l1_sz_sum+=ltt_l1_sz
        ltt_l2_sz_sum+=ltt_l2_sz
        ltt_total_sz_sum+=ltt_total_sz
    ltt_l1_risk_val_avg = ltt_l1_risk_val_sum/num_iterations
    ltt_l2_risk_val_avg = ltt_l2_risk_val_sum/num_iterations
    ltt_l1_sz_avg= ltt_l1_sz_sum/num_iterations
    ltt_l2_sz_avg= ltt_l2_sz_sum/num_iterations
    ltt_total_sz_avg = ltt_total_sz_sum/num_iterations
    print('{},{},{},{},{},{}'.format(alpha, ltt_total_sz_avg,ltt_l1_sz_avg, ltt_l2_sz_avg, ltt_l1_risk_val_avg, ltt_l2_risk_val_avg ))