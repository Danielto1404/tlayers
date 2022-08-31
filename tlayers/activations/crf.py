from typing import Optional

import torch
from torch import nn


class CRF(nn.Module):
    __REDUCTIONS__ = ("mean", "sum")

    def __init__(
            self,
            num_tags: int,
            batch_first: bool = False
    ):
        super(CRF, self).__init__()

        assert num_tags > 1, f"Number of tags must be positive number."

        self.num_tags = num_tags
        self.batch_first = batch_first

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between `-0.1` and `0.1`.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
            Args:

                emissions (`~torch.Tensor`): Emission score tensor of size
                    ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                    ``(batch_size, seq_length, num_tags)`` otherwise.
                tags (`~torch.LongTensor`): Sequence of tags tensor of size
                    ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                    ``(batch_size, seq_length)`` otherwise.
                mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                    if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
                reduction: Specifies  the reduction to apply to the output:
                    ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                    ``sum``: the output will be summed over batches. ``mean``: the output will be
                    averaged over batches. ``token_mean``: the output will be averaged over tokens.
            Returns:
                `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
                reduction is ``none``, ``()`` otherwise.
        """
        assert reduction in self.__REDUCTIONS__, \
            f"Got invalid reduction: {reduction}. Possible reduction options: {self.__REDUCTIONS__}"

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

    def _compute_score(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: torch.ByteTensor
    ) -> torch.Tensor:
        """

        :param emissions: seq x batch x num_tags
        :param tags: seq x batch
        :param mask: seq x batch
        :return:
        """

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'


if __name__ == '__main__':
    crf = CRF(10)
    print(crf)
