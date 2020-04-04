import numpy as np

def sample_from(sources, weights, req = 100):
    inds = np.random.choice(len(weights), req)

    sampled_weights = weights[inds]
    sampled_data = [cur_source[inds] for cur_source in sources]

    return sampled_data, sampled_weights        

# draw a certain number of events from 'components'. If 'equalize_fraction' is set to 'False', the
# original proportions of the individual components are kept.
def sample_from_components(components, weights, batch_size = 1000, sampling_pars = {}):
    sampling_pars.setdefault("sampling_fractions", None)

    # Note: this effectively transposes the nested lists such that the iterations become easier
    sources = list(map(list, zip(*components)))        

    # first, compute the number of events available for each signal / background component ...
    nevents = [len(cur) for cur in weights]
    total_nevents = np.sum(nevents)

    # ... and also the SOW for each
    if sampling_pars["sampling_fractions"] is not None:
        # make sure they sum up to one
        sampling_pars["sampling_fractions"] /= np.sum(sampling_pars["sampling_fractions"])

        # pretend that each component came with an equal SOW to start with
        SOWs = np.array(sampling_pars["sampling_fractions"])
    else:
        # keep the original proportions
        SOWs = [np.sum(cur) for cur in weights]
        total_SOW = np.sum(SOWs)
        SOWs /= total_SOW # normalize total SOW to 1

    total_SOW = 0

    # check if a manual override is provided for the number of samples drawn from each signal / background component
    samplinglengths = [1.0 for cur_source in sources]
    if "sampling_lengths" in sampling_pars:
        samplinglengths = sampling_pars["sampling_lengths"]

    # now, compute the number of events that should be sampled from each signal / background component:
    # sample them in the same proportions with which they appear in the training dataset ...
    sampled_data = []
    sampled_weights = []
    for cur_source, cur_weights, cur_nevents, cur_samplinglength in zip(sources, weights, nevents, samplinglengths):
        cur_sampled_data, cur_sampled_weights = sample_from(cur_source, cur_weights, req = int(cur_samplinglength * batch_size / len(sources)))
        sampled_data.append(cur_sampled_data)
        sampled_weights.append(cur_sampled_weights)
        total_SOW += np.sum(cur_sampled_weights)
        
    # ... and normalize them such that their SOWs are in the correct relation to each other
    for cur, cur_SOW in enumerate(SOWs):
        cur_sum = np.sum(sampled_weights[cur])
        if cur_sum > 0:
            sampled_weights[cur] *= cur_SOW / cur_sum # each batch will have a total SOW of 1

    # transpose them back for easy concatenation
    sampled_sources = list(map(list, zip(*sampled_data)))

    # perform the concatenation ...
    sampled = [np.concatenate(cur, axis = 0) for cur in sampled_sources]
    sampled_weights = np.concatenate(sampled_weights, axis = 0)
    
    sampled_weights *= batch_size / 10.0 # perform some scaling of the weights

    # ... and return
    return sampled, sampled_weights

def sample_from_TrainingSamples(samples, size, sampling_pars = {}):
    
    # extract the individual contents and then call the underlying sampling function
    data = [cur_sample.data for cur_sample in samples]
    nuis = [cur_sample.nuis for cur_sample in samples]
    weights = [cur_sample.weights for cur_sample in samples]
    labels = [cur_sample.labels for cur_sample in samples]

    return sample_from_components([data, nuis, labels], weights, batch_size = size, sampling_pars = sampling_pars)
    
def all(samples):

    data = np.concatenate([cur_sample.data for cur_sample in samples], axis = 0)
    nuis = np.concatenate([cur_sample.nuis for cur_sample in samples], axis = 0)
    weights = np.concatenate([cur_sample.weights for cur_sample in samples], axis = 0)
    labels = np.concatenate([cur_sample.labels for cur_sample in samples], axis = 0)

    return (data, nuis, labels), weights

