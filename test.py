import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in 'Attention Is All You Need' (Vaswani et al., 2017)
    
    This module splits the input into multiple heads, applies scaled dot-product attention 
    independently to each head, and then recombines the results. This allows the model to
    jointly attend to information from different representation subspaces at different positions.
    
    Args:
        embed_size (int): Dimensionality of the input embeddings
        heads (int): Number of attention heads
    """
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Ensure embed_size is divisible by heads for even splitting
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"
        
        # Linear projections for values, keys, and queries
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        
        # Final linear projection after attention
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query):
        """
        Forward pass for the Multi-Head Attention module
        
        Args:
            values (torch.Tensor): Value tensors of shape [batch_size, seq_len, embed_size]
            keys (torch.Tensor): Key tensors of shape [batch_size, seq_len, embed_size]
            query (torch.Tensor): Query tensors of shape [batch_size, seq_len, embed_size]
            
        Returns:
            torch.Tensor: Output after applying multi-head attention, shape [batch_size, seq_len, embed_size]
        """
        # Get batch size and sequence length from query shape
        batch_size, seq_len, _ = query.shape
        
        # 1. Linear projections and reshape for multi-head processing
        # Transform from [batch_size, seq_len, embed_size] to [batch_size, heads, seq_len, head_dim]
        values = self.values(values).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(keys).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        queries = self.queries(query).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # 2. Calculate attention scores
        # Scaled dot-product attention: Q·K^T/sqrt(d_k)
        # Shape: [batch_size, heads, seq_len, seq_len]
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 3. Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 4. Apply attention weights to values
        # Shape: [batch_size, heads, seq_len, head_dim]
        attention_output = torch.matmul(attention_weights, values)
        
        # 5. Reshape back to original dimensions
        # [batch_size, heads, seq_len, head_dim] -> [batch_size, seq_len, embed_size]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
        
        # 6. Final linear projection
        output = self.fc_out(attention_output)
        
        return output


# Example usage with dummy data
if __name__ == "__main__":
    # Set dimensions
    batch_size, seq_len, embed_size, heads = 7, 100, 512, 8
    
    # Create random input tensors
    dummy_values = torch.randn(batch_size, seq_len, embed_size)
    dummy_keys = torch.randn(batch_size, seq_len, embed_size)
    dummy_queries = torch.randn(batch_size, seq_len, embed_size)
    
    # Initialize the MultiHeadAttention module
    multi_head_attn = MultiHeadAttention(embed_size=embed_size, heads=heads)
    
    # Forward pass
    output = multi_head_attn(dummy_values, dummy_keys, dummy_queries)
    print("Output shape:", output.shape)  # Expected: [batch_size, seq_len, embed_size]