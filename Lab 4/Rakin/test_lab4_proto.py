import torch
from lab4_proto import *


# Test case 2: Single batch, multiple time steps
output2 = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]])  # B x T x C
decoded_strings2 = greedyDecoder(output2)
print("Utfall: ", decoded_strings2)
# Expected output: ['e', 'a']
assert decoded_strings2 == ['e', 'a']

# Test case 3: Multiple batches, multiple time steps
output3 = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
                        [[0.2, 0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3, 0.2]]])  # B x T x C
decoded_strings3 = greedyDecoder(output3)
# Expected output: ['e', 'a', 'e', 'a']
assert decoded_strings3 == ['e', 'a', 'e', 'a']

print("All test cases passed!")
