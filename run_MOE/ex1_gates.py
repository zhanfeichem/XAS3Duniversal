import torch
import torch.nn.functional as F

class MOEModel:
    def __init__(self, num_experts, k):
        self.num_experts = num_experts
        self.k = k

    def softmax(self, x):
        return F.softmax(x, dim=1)

    def forward(self, logits):
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices,top_k_gates )
        return gates,top_k_logits, top_k_indices, top_k_gates

# Example usage
batch_size = 4
num_experts = 6
k = 2

# Create a random logits tensor with shape [batch_size, num_experts]
logits = torch.randn(batch_size, num_experts)

# Instantiate the model
model = MOEModel(num_experts, k)

# Forward pass through the model
gates,top_k_logits, top_k_indices, top_k_gates = model.forward(logits)

# Print shapes to verify dimensions
print("Logits shape:", logits.shape)
print("Top K Logits shape:", top_k_logits.shape)
print("Top K Indices shape:", top_k_indices.shape)
print("Top K Gates shape:", top_k_gates.shape)
print("Gates shape:",gates.shape)


print("PAUSE")