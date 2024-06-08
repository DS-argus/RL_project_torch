import numpy as np
import torch

def evaluate(recommender, env, check_movies=False, top_k=False, length=False):
    # episodic reward
    mean_precision = 0
    mean_ndcg = 0

    # episodic reward
    episode_reward = 0
    steps = 0
    q_loss1 = 0
    q_loss2 = 0
    countl = 0
    correct_list = []

    # Environment     
    user_id, items_ids, done = env.reset()
    while not done:
        # Observe current state & Find action         
        # Embedding         
        user_eb = recommender.embedding_network.user_embedding(torch.tensor([user_id], dtype=torch.long)).detach().numpy()
        items_eb = recommender.embedding_network.movie_embedding(torch.tensor(items_ids, dtype=torch.long)).detach().numpy()
        # SRM state         
        state = recommender.srm_ave([
            torch.tensor(user_eb, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(items_eb, dtype=torch.float32).unsqueeze(0)
        ])
        # Action (ranking score)         
        action = recommender.actor.network(state).detach().numpy()
        # Item         
        recommended_item = recommender.recommend_item(
            action, env.recommended_items, top_k=top_k)

        next_items_ids, reward, done, _ = env.step(
            recommended_item, top_k=top_k)
        
        if countl < length:
            countl += 1
            correct_list.append(reward)
            if done:
                dcg, idcg = calculate_ndcg(
                    correct_list, [1 for _ in range(len(correct_list))])
                mean_ndcg += dcg/idcg
                print("mean_ndcg :", mean_ndcg)

            # precision
            correct_list1 = [1 if r > 0 else 0 for r in correct_list]
            correct_num = length - correct_list1.count(0)
            mean_precision += correct_num / length

        items_ids = next_items_ids
        episode_reward += reward
        steps += 1

    return mean_precision, mean_ndcg, reward


def calculate_ndcg(rel, irel):
    dcg = 0
    idcg = 0
    rel = [1 if r > 0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r) / np.log2(i + 2)
        idcg += (ir) / np.log2(i + 2)
    return dcg, idcg
