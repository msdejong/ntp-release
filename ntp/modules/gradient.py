"""Calculate gradients given a selection from a dictionary of loss functions"""

import copy

import numpy as np
import tensorflow as tf

from ntp.modules.kb import kb2nkb
from ntp.modules.prover import prove, kb_form
from ntp.modules.nunify import representation_match

def auto_gradient(loss_function, goal, target, emb, kb_ids, vocab, goal_struct, batch_mask, l2, k_max, max_depth, epsilon, loss_parameters):
    """Automatically calculate gradient using gradient tape. Can be significantly slower than calculating manually"""

    with tf.GradientTape() as tape:
        kb_goal = copy.deepcopy(kb_ids)
        kb_goal['goal'] = [[row.numpy() for row in goal]]
        nkb = kb2nkb(kb_goal, emb)

        goal = [{'struct': 'goal', 'atom': 0, 'symbol': i} for i in range(goal.ndim)]

        proofs = prove(nkb, goal, goal_struct, batch_mask,
                    k_max=k_max, max_depth=max_depth, vocab=vocab)

        loss = loss_function(proofs, target, epsilon, loss_parameters)
        loss += tf.nn.l2_loss(emb) * l2
        grad = tape.gradient(loss, emb)

    return grad

def standard_loss(proofs, target, epsilon, loss_parameters):
    """Loss given by score of best proof"""
    success = tf.squeeze(retrieve_top_k(proofs))
    loss = loss_from_proof(success, target, epsilon)
    return loss

def all_type_loss(proofs, target, epsilon, loss_parameters):
    """Loss given by sum of scores of best proofs of each type"""
    loss = 0
    for proof in proofs:
        success = tf.squeeze(retrieve_top_k([proof]))
        temp_loss = loss_from_proof(success, target, epsilon)
        loss += temp_loss
    return loss * 2

def top_k_loss(proofs, target, epsilon, loss_parameters):
    """Loss given by sum of scores of top-k proofs"""
    k = loss_parameters["k"]
    success = retrieve_top_k(proofs, k)
    success_split = tf.split(success, k, axis=-1)
    loss = 0
    for sub_success in success_split:
        sub_success = tf.squeeze(sub_success)
        temp_loss = loss_from_proof(sub_success, target, epsilon)
        loss += temp_loss
    return loss

def top_k_all_type_loss(proofs, target, epsilon, loss_parameters):
    """Loss given by sum of scores of top-k proofs for each proof path"""
    k = loss_parameters["k"]
    loss = 0
    for proof in proofs:
        success = retrieve_top_k([proof], k)
        success_split = tf.split(success, k, axis=-1)
        for sub_success in success_split:
            sub_success = tf.squeeze(sub_success)
            temp_loss = loss_from_proof(sub_success, target, epsilon)
            loss += temp_loss
    return loss

def top_k_all_anneal_loss(proofs, target, epsilon, loss_parameters):
    """Choose loss function based on epoch in training, annealing from all-path to normal or top-k to top-1"""
    epoch = loss_parameters["epoch"]    
    regime_thresholds = loss_parameters["regime_thresholds"]
    regime_types = loss_parameters["regime_types"]
    k_values = loss_parameters["k_values"]

    regime = np.digitize(epoch, regime_thresholds) - 1

    new_parameters = {"k": k_values[regime]}
    regime_function = loss_dict[regime_types[regime]] 


    return regime_function(proofs, target, epsilon, new_parameters)

def top_k_kernel_anneal_loss(proofs, target, epsilon, loss_parameters):
    """Take weighted average over top-k proofs, placing more weight on the best proof as training progresses"""
    epoch = loss_parameters["epoch"]

    k = loss_parameters["k"]
    loss = 0
    successes = []
    for proof in proofs:
        success = retrieve_top_k([proof], k)
        successes.append(success)
    
    success = tf.concat(successes, axis=1)
    success = tf.nn.top_k(success, success.get_shape()[-1])[0]
    success_split = tf.split(success, success.get_shape()[-1], axis=-1)
    for j, sub_success in enumerate(success_split):
        sub_success = tf.squeeze(sub_success)
        temp_loss = loss_from_proof(sub_success, target, epsilon)
        loss += (50/(50 + epoch))**j * temp_loss

    return loss

def top_k_score_anneal_loss(proofs, target, epsilon, loss_parameters):
    """Choose loss function based on score of best proof, using more proofs as the certainty of the model decreases"""    

    success = tf.squeeze(retrieve_top_k(proofs))
    regime_thresholds = loss_parameters["regime_thresholds"]

    regimes = np.digitize(success, regime_thresholds) - 1
    k_values = [loss_parameters["k_values"][regimes[i]] for i in range(len(regimes))]

    max_k = np.max(k_values)

    if max_k > 1:
        success = retrieve_top_k(proofs, k=max_k)
    else: 
        return standard_loss(proofs, target, epsilon, loss_parameters)

    n_goals = success.get_shape()[0]
    mask = np.zeros((n_goals, max_k))

    for i in range(n_goals):
        mask[i, :k_values[i]] += 1

    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    x = success
    z = tf.expand_dims(target, axis=-1)
    loss_tensor = -z * tf.log(tf.clip_by_value(x, epsilon, 1.0)) - \
        (1 - z) * tf.log(tf.clip_by_value(1 - x, epsilon, 1.0))

    loss = mask * loss_tensor

    total_loss = tf.reduce_sum(loss)
    
    return total_loss

        
def top_k_score_anneal_all_loss(proofs, target, epsilon, loss_parameters):
    """Choose loss function based on score of best proof, using more proofs and of more types as the certainty of the model decreases"""    
    
    success = tf.squeeze(retrieve_top_k(proofs))
    regime_thresholds = loss_parameters["regime_thresholds"]

    regimes = np.digitize(success, regime_thresholds) - 1
    k_values = [loss_parameters["k_values"][regimes[i]] for i in range(len(regimes))]
    all_type = [loss_parameters["all_type"][regimes[i]] for i in range(len(regimes))]

    max_k = loss_parameters["max_k"]

    success = retrieve_top_k(proofs, k=max_k)
    proof_successes = [retrieve_top_k([proof], k=max_k) for proof in proofs]
    
 
    n_goals = success.get_shape()[0]

    k_mask = np.zeros((n_goals, max_k))
    for i in range(n_goals):
        k_mask[i, :k_values[i]] += 1

    type_mask = tf.expand_dims(tf.convert_to_tensor(all_type, dtype=tf.float32), axis=-1)
    base_mask = k_mask * (1 - type_mask)
    all_mask = k_mask * type_mask
    
    base_mask = tf.convert_to_tensor(base_mask, dtype=tf.float32)
    all_mask = tf.convert_to_tensor(all_mask, dtype=tf.float32)

    x = success
    z = tf.expand_dims(target, axis=-1)
    loss_tensor = -z * tf.log(tf.clip_by_value(x, epsilon, 1.0)) - \
        (1 - z) * tf.log(tf.clip_by_value(1 - x, epsilon, 1.0))

    loss = base_mask * loss_tensor

    total_loss = tf.reduce_sum(loss)

    for proof_success in proof_successes:
        x = proof_success
        loss_tensor = -z * tf.log(tf.clip_by_value(x, epsilon, 1.0)) - \
            (1 - z) * tf.log(tf.clip_by_value(1 - x, epsilon, 1.0))

        loss = all_mask * loss_tensor

        total_loss += tf.reduce_sum(loss)

    return total_loss


loss_dict = {
    "standard": standard_loss,
    "all_type": all_type_loss,
    "top_k": top_k_loss,
    "top_k_all_type": top_k_all_type_loss,
    "top_k_all_anneal": top_k_all_anneal_loss,
    "top_k_kernel_anneal": top_k_kernel_anneal_loss,
    "top_k_score_anneal":top_k_score_anneal_loss,
    "top_k_score_anneal_all":top_k_score_anneal_all_loss
}


def retrieve_top_k(proofs, k=1):
    """Given a list of proofs, find the k overall best proofs"""
    tensors = [proof["SUCCESS"] for proof in proofs]

    for i, tensor in enumerate(tensors):
        proof_shape = tensor.get_shape()
        num_goals = proof_shape[0]
        success_per_proof = tf.reshape(tensor, [num_goals, -1])
        tensors[i] = success_per_proof
    if len(tensors) > 1:
        success_per_proof = tf.concat(tensors, 1)
    else:
        success_per_proof = tensors[0]
    
    top_k = tf.nn.top_k(success_per_proof, k)

    return top_k[0]

def loss_from_proof(success, target, epsilon):
    """Given success of a proof, a target and epsilon, calculate loss from just that proof"""
    x = success
    z = target
    loss = -z * tf.log(tf.clip_by_value(x, epsilon, 1.0)) - \
        (1 - z) * tf.log(tf.clip_by_value(1 - x, epsilon, 1.0))
    loss = tf.reduce_sum(loss)
    return loss


##### section for calculating gradient manually, not in use currently


def manual_gradient(goal, target, emb, kb_ids, vocab, goal_struct, batch_mask, l2, k_max, max_depth, n_facts):
    """Use simple structure of model to calculate gradient manually given goal, training facts, and embedding. """

    # Get nkb that includes 'fake' struct consisting of goal
    kb_goal = copy.deepcopy(kb_ids)
    kb_goal['goal'] = [[row.numpy() for row in goal]]
    nkb = kb2nkb(kb_goal, emb)

    goal = [{'struct': 'goal', 'atom': 0, 'symbol': i} for i in range(len(goal_struct[0]))]

    proofs = prove(nkb, goal, goal_struct, batch_mask,
                   k_max=k_max, max_depth=max_depth, vocab=vocab)

    emb_np = emb.numpy()
    shape = emb_np.shape

    grad = calculate_gradient(proofs, nkb, kb_goal, emb, target.numpy(), shape, n_facts)
    grad = tf.convert_to_tensor(grad, dtype=tf.float32)
    grad += l2 * emb

    return grad

def calculate_gradient(proofs, nkb, kb_ids, emb, targets, shape, n_facts):
    """
    Calculates the gradient for a particular proof manually from the array that stores active (worst) unifications per proof. 
    Args:
        proofs: list of proofs, each proof is a dict with information about a proof type
        emb: embedding matrix
        targets: 1/0 depending on whether goal is corrupted or not
        shape: shape of emb (and therefore gradient)
        n_facts: facts in kb
        kb_ids: Contains facts of kb, with symbol ids
        goal_struct: struct of goal, i.e. ('p', 'c')
    Returns:
        gradient
    """

    proof_successes = [proof['SUCCESS'].numpy() for proof in proofs]
    success_reshaped = [np.reshape(
        success, (success.shape[0], -1)) for success in proof_successes]
    
    success_max = np.array([np.max(success, axis=1) for success in success_reshaped])
    best_proof_indices = np.argmax(success_max, axis=0)

    n_goal = len(best_proof_indices)

    unifications = [proof['UNIFICATIONS'] for proof in proofs]

    indices_a = []
    indices_b = []

    goal_shapes = []

    for i in range(n_goal):

        best_proof_index = best_proof_indices[i]
        best_indices_reshaped = np.argmax(success_reshaped[best_proof_index][i])
        best_indices = np.unravel_index(best_indices_reshaped, proof_successes[best_proof_index].shape[1:])

        best_indices = [i] + [value for value in best_indices]

        goal_shapes.append(len(unifications[best_proof_indices[i]]))

        goal_unifications = unifications[best_proof_indices[i]] 
    
        for unification in goal_unifications:

            position_a = unification[2][0]
            position_b = unification[2][1]

            index_a = best_indices[position_a]
            index_b = best_indices[position_b]

            indices_a.append(kb_form(unification[0], kb_ids)[index_a])
            indices_b.append(kb_form(unification[1], kb_ids)[index_b])
        

    indices_a = tf.constant(indices_a)
    indices_b = tf.constant(indices_b)

    tensor_a = tf.nn.embedding_lookup(emb, indices_a)
    tensor_b = tf.nn.embedding_lookup(emb, indices_b)

    tensor_dif = tensor_b - tensor_a
    scores = tf.exp(-tf.norm(tensor_dif, axis=1))
    
    reshaped_scores = tf.split(scores, goal_shapes, axis=0)

    cumulative_shape = [0]
    for i in range(len(goal_shapes) - 1):
        cumulative_shape.append(cumulative_shape[-1] + goal_shapes[i])

    argmin_scores = [np.argmin(scores) for scores in reshaped_scores]


    dif_indices = [cumulative_shape[i] + argmin_scores[i] for i in range(len(goal_shapes))]

    tensor_dif = tf.gather(tensor_dif, dif_indices)
    
    reshaped_indices = tf.stack((indices_a, indices_b), axis=1)
    reshaped_indices = tf.split(reshaped_indices, goal_shapes, axis=0)
    reshaped_indices = [value.numpy() for value in reshaped_indices]

    active_indices = [reshaped_indices[i][argmin_scores[i]] for i in range(len(argmin_scores))]
    active_scores = [tf.reduce_min(scores) for scores in reshaped_scores]

    active_scores = tf.expand_dims(active_scores, axis=1)
    targets = tf.expand_dims(targets, axis=1)
    
    gradient_values = targets * tensor_dif / tf.log(active_scores) + \
            (1 - targets) * active_scores / (1 - active_scores) * -tensor_dif / tf.log(active_scores)

    gradient_values = gradient_values.numpy()

    gradient = np.zeros(shape)


    for i in range(n_goal):
        gradient[active_indices[i][0]] += gradient_values[i]
        gradient[active_indices[i][1]] -= gradient_values[i] 

    return gradient
