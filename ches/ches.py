import torch


def compute_ches_scores(preferred_hidden_embeddings: torch.Tensor, dispreferred_hidden_embeddings: torch.Tensor,
                        preferred_last_prompt_token_indices: torch.Tensor, dispreferred_last_prompt_token_indices: torch.Tensor,
                        length_normalize: bool = False):
    """
    Compute CHES scores based on the hidden embeddings of preferred and dispreferred responses.
    The preferred hidden embeddings are the embeddings produced when the model is given both the prompt and the preferred response, i.e. $(x, y^+)$,
    and the dispreferred hidden embeddings are those produced when the model is given the prompt and the dispreferred response, i.e. $(x, y^-)$.
    @param preferred_hidden_embeddings: Tensor of shape (batch size, (padded) prompt + preferred sequence length, embedding dimension).
    @param dispreferred_hidden_embeddings: Tensor of shape (batch size, (padded) prompt + dispreferred sequence length, embedding dimension).
    @param preferred_last_prompt_token_indices: Tensor of shape (batch size,) containing the indices of the last prompt token in the sequence used
    to compute the preferred hidden embeddings.
    @param dispreferred_last_prompt_token_indices: Tensor of shape (batch size,) containing the indices of the last prompt token in the sequence used
    to compute the dispreferred hidden embeddings.
    @param length_normalize: If True, compute the length-normalized CHES scores.
    @return: Tensor of shape (batch size,) containing the CHES scores.
    """
    # Zero out prompt embeddings except last one
    pref_mask = torch.arange(preferred_hidden_embeddings.size(1),
                             device=preferred_hidden_embeddings.device).expand(preferred_hidden_embeddings.size(0),
                                                                               preferred_hidden_embeddings.size(1))
    pref_mask = pref_mask >= preferred_last_prompt_token_indices.unsqueeze(1)
    preferred_hidden_embeddings = preferred_hidden_embeddings * pref_mask.unsqueeze(2)

    dispref_mask = torch.arange(dispreferred_hidden_embeddings.size(1),
                                device=dispreferred_hidden_embeddings.device).expand(dispreferred_hidden_embeddings.size(0),
                                                                                     dispreferred_hidden_embeddings.size(1))
    dispref_mask = dispref_mask >= dispreferred_last_prompt_token_indices.unsqueeze(1)
    dispreferred_hidden_embeddings = dispreferred_hidden_embeddings * dispref_mask.unsqueeze(2)

    # Remove last token of a response, whose hidden embedding does not take part when computing CHES scores (this is usually the hidden embedding of the EOS token)
    preferred_hidden_embeddings = preferred_hidden_embeddings[:, :-1]
    dispreferred_hidden_embeddings = dispreferred_hidden_embeddings[:, :-1]

    sum_preferred_embeddings = preferred_hidden_embeddings.sum(dim=1)
    sum_dispreferred_embeddings = dispreferred_hidden_embeddings.sum(dim=1)

    if not length_normalize:
        return (sum_preferred_embeddings * sum_dispreferred_embeddings).sum(dim=1) - torch.norm(sum_preferred_embeddings, dim=1) ** 2

    preferred_lengths = preferred_hidden_embeddings.shape[1] - preferred_last_prompt_token_indices
    dispreferred_lengths = dispreferred_hidden_embeddings.shape[1] - dispreferred_last_prompt_token_indices

    pref_dispref = (sum_preferred_embeddings * sum_dispreferred_embeddings).sum(dim=1) / (preferred_lengths * dispreferred_lengths)
    pref_only = torch.norm(sum_preferred_embeddings, dim=1) ** 2 / (preferred_lengths ** 2)
    return pref_dispref - pref_only
