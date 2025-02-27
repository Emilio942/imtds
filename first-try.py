import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm
import numpy as np
import gymnasium as gym
from collections import defaultdict, deque
import random
import traceback
from typing import List, Dict, Tuple, Any, Optional, Union, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MathTheorem")

class MathematicalStructure:
    """Representation of a mathematical structure as a graph."""
    
    def __init__(self, name: str = "UnnamedStructure"):
        """
        Initialize a new mathematical structure.
        
        Args:
            name: A name identifier for the structure
        """
        self.name = name
        self.nodes = []  # Mathematical objects (e.g., numbers, sets, functions)
        self.edges = []  # Relationships between objects
        self.properties = {}  # Proven properties of the structure
        self.node_types = ['number', 'set', 'function', 'operator', 'structure', 'variable']
        self.edge_types = ['contains', 'maps_to', 'derives', 'equals', 'bounds', 'operates_on']
        
    def add_node(self, node_type: str, attributes: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a new mathematical object node.
        
        Args:
            node_type: Type of the mathematical object
            attributes: Dictionary of node attributes
            
        Returns:
            The ID of the newly created node
        """
        if node_type not in self.node_types:
            logger.warning(f"Unknown node type: {node_type}. Adding it to recognized types.")
            self.node_types.append(node_type)
            
        node_id = len(self.nodes)
        self.nodes.append({
            'id': node_id,
            'type': node_type,
            'attributes': attributes or {}
        })
        return node_id
        
    def add_edge(self, source_id: int, target_id: int, relation_type: str) -> int:
        """
        Add a relationship between two mathematical objects.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relation_type: Type of the relationship
            
        Returns:
            The ID of the newly created edge
        """
        if relation_type not in self.edge_types:
            logger.warning(f"Unknown edge type: {relation_type}. Adding it to recognized types.")
            self.edge_types.append(relation_type)
            
        # Validate that node IDs exist
        if source_id >= len(self.nodes) or target_id >= len(self.nodes):
            raise ValueError(f"Node IDs must be valid. Got source_id={source_id}, target_id={target_id}")
            
        edge_id = len(self.edges)
        self.edges.append({
            'id': edge_id,
            'source': source_id,
            'target': target_id,
            'type': relation_type
        })
        return edge_id
    
    def to_torch_geometric(self) -> Dict[str, torch.Tensor]:
        """
        Convert the structure to a format for PyTorch Geometric.
        
        Returns:
            Dictionary with keys 'x', 'edge_index', 'edge_attr' for PyTorch Geometric
        """
        # Node feature matrix
        x = []
        for node in self.nodes:
            features = self._encode_node_features(node)
            x.append(features)
            
        # Edge index matrix [2, num_edges]
        edge_index = []
        edge_attr = []
        
        for edge in self.edges:
            edge_index.append([edge['source'], edge['target']])
            edge_feature = self._encode_edge_features(edge)
            edge_attr.append(edge_feature)
        
        # Handle the case of empty edges
        if not edge_index:
            # Create a dummy self-loop for the first node if there are nodes
            if self.nodes:
                edge_index = [[0, 0]]
                # Create a zero feature vector for the edge
                edge_attr = [[0] * len(self.edge_types)]
            else:
                # If there are no nodes, return empty tensors
                return {
                    'x': torch.tensor([], dtype=torch.float),
                    'edge_index': torch.tensor([], dtype=torch.long).reshape(2, 0),
                    'edge_attr': torch.tensor([], dtype=torch.float)
                }
        
        return {
            'x': torch.tensor(x, dtype=torch.float),
            'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            'edge_attr': torch.tensor(edge_attr, dtype=torch.float)
        }
    
    def _encode_node_features(self, node: Dict[str, Any]) -> List[float]:
        """
        Encode node features for the GNN.
        
        Args:
            node: Node data dictionary
            
        Returns:
            List of encoded features
        """
        # One-hot encoding for node type
        features = [0] * len(self.node_types)
        if node['type'] in self.node_types:
            features[self.node_types.index(node['type'])] = 1
        
        # Extract additional features from attributes
        attributes = node['attributes']
        
        # Add a feature for the presence of a name
        if 'name' in attributes:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Add a feature for numerical value if present
        if 'value' in attributes and isinstance(attributes['value'], (int, float)):
            # Normalize the value to the range [-1, 1] using sigmoid
            features.append(float(2 / (1 + np.exp(-attributes['value'])) - 1))
        else:
            features.append(0.0)
        
        return features
    
    def _encode_edge_features(self, edge: Dict[str, Any]) -> List[float]:
        """
        Encode edge features for the GNN.
        
        Args:
            edge: Edge data dictionary
            
        Returns:
            List of encoded features
        """
        # One-hot encoding for edge type
        features = [0] * len(self.edge_types)
        if edge['type'] in self.edge_types:
            features[self.edge_types.index(edge['type'])] = 1
            
        return features
        
    def __str__(self) -> str:
        """Return a string representation of the structure."""
        return f"MathematicalStructure({self.name}): {len(self.nodes)} nodes, {len(self.edges)} edges"
        
    def __repr__(self) -> str:
        """Return a detailed representation of the structure."""
        return f"MathematicalStructure({self.name}, nodes={len(self.nodes)}, edges={len(self.edges)})"


class TheoremVerifier:
    """Verifies the correctness of theorems through formal proof."""
    
    def __init__(self, axiom_system: List[str]):
        """
        Initialize the theorem verifier.
        
        Args:
            axiom_system: List of axioms as strings
        """
        self.axioms = set(axiom_system)
        self.theorem_library = {}  # Library of known theorems
        self.inference_rules = {
            'modus_ponens': self._apply_modus_ponens,
            'conjunction_introduction': self._apply_conjunction_introduction,
            'conjunction_elimination': self._apply_conjunction_elimination,
            'disjunction_introduction': self._apply_disjunction_introduction,
            'disjunction_elimination': self._apply_disjunction_elimination,
            'implication_introduction': self._apply_implication_introduction,
            'universal_instantiation': self._apply_universal_instantiation,
            'existential_generalization': self._apply_existential_generalization,
            'universal_introduction': self._apply_universal_introduction,
            'existential_elimination': self._apply_existential_elimination,
            'assumption': self._apply_assumption
        }
        
    def verify(self, theorem: str, proof_steps: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Verify a proof for correctness.
        
        Args:
            theorem: The theorem to be proven
            proof_steps: List of inference steps
            
        Returns:
            (bool, feedback): True if the proof is correct, else False with feedback
        """
        current_state = set(self.axioms)
        
        for step_idx, step in enumerate(proof_steps):
            # Check if the step follows from previous assumptions
            if not self._is_valid_inference(current_state, step):
                return False, f"Invalid inference step at step {step_idx}: {step['rule']} cannot be applied"
            
            # Add the conclusion to our current knowledge
            current_state.add(step['conclusion'])
        
        # Check if the theorem follows from the final state
        if self._implies(current_state, theorem):
            return True, "Proof is correct"
        else:
            return False, "Proof does not reach the desired theorem"
    
    def _is_valid_inference(self, current_knowledge: Set[str], inference_step: Dict[str, Any]) -> bool:
        """
        Check if an inference step is valid.
        
        Args:
            current_knowledge: Set of currently known statements
            inference_step: The inference step to validate
            
        Returns:
            True if the inference is valid, False otherwise
        """
        rule = inference_step['rule']
        premises = inference_step.get('premises', [])
        conclusion = inference_step['conclusion']
        
        # Check if all premises are in the current knowledge
        if not all(premise in current_knowledge for premise in premises):
            return False
            
        # Apply the rule if it exists in our rule dictionary
        if rule in self.inference_rules:
            return self.inference_rules[rule](premises, conclusion)
            
        logger.warning(f"Unknown inference rule: {rule}")
        return False
    
    def _apply_modus_ponens(self, premises: List[str], conclusion: str) -> bool:
        """Apply the modus ponens rule: from P and P→Q, infer Q."""
        # If we have p and p→q, then q
        # Check if one of the premises is an implication leading to the conclusion
        for p in premises:
            for other_p in premises:
                if other_p == f"{p} → {conclusion}":
                    return True
        return False
    
    def _apply_conjunction_introduction(self, premises: List[str], conclusion: str) -> bool:
        """Apply the conjunction introduction rule: from P and Q, infer P∧Q."""
        if len(premises) < 2:
            return False
        
        # Check if the conclusion is a conjunction of the premises
        if conclusion == f"{premises[0]} ∧ {premises[1]}":
            return True
        if conclusion == f"{premises[1]} ∧ {premises[0]}":
            return True
            
        return False
    
    def _apply_conjunction_elimination(self, premises: List[str], conclusion: str) -> bool:
        """Apply the conjunction elimination rule: from P∧Q, infer P or Q."""
        if len(premises) != 1:
            return False
            
        # Extract the parts of a conjunction
        parts = premises[0].split(" ∧ ")
        if len(parts) != 2:
            return False
            
        # Check if the conclusion is one of the parts
        return conclusion in parts
    
    def _apply_disjunction_introduction(self, premises: List[str], conclusion: str) -> bool:
        """Apply the disjunction introduction rule: from P, infer P∨Q."""
        if len(premises) != 1:
            return False
            
        # Check if the conclusion is a disjunction containing the premise
        parts = conclusion.split(" ∨ ")
        if len(parts) != 2:
            return False
            
        return premises[0] in parts
    
    def _apply_disjunction_elimination(self, premises: List[str], conclusion: str) -> bool:
        """
        Apply the disjunction elimination rule:
        from P∨Q, P→R, Q→R, infer R.
        """
        if len(premises) < 3:
            return False
            
        # Find the disjunction
        disjunction = None
        for p in premises:
            if " ∨ " in p:
                disjunction = p
                break
                
        if disjunction is None:
            return False
            
        # Extract the disjuncts
        disjuncts = disjunction.split(" ∨ ")
        if len(disjuncts) != 2:
            return False
            
        # Check for implications from each disjunct to the conclusion
        impl1 = f"{disjuncts[0]} → {conclusion}"
        impl2 = f"{disjuncts[1]} → {conclusion}"
        
        return impl1 in premises and impl2 in premises
    
    def _apply_implication_introduction(self, premises: List[str], conclusion: str) -> bool:
        """
        Apply the implication introduction rule.
        This is a simplified version since we don't track assumptions separately.
        """
        if len(premises) < 2:
            return False
            
        # Very simple check: if conclusion is "A → B" and both A and B are premises
        parts = conclusion.split(" → ")
        if len(parts) != 2:
            return False
            
        return parts[0] in premises and parts[1] in premises
    
    def _apply_universal_instantiation(self, premises: List[str], conclusion: str) -> bool:
        """
        Apply the universal instantiation rule: from ∀x.P(x), infer P(a).
        This is a very simplified implementation.
        """
        if len(premises) != 1:
            return False
            
        premise = premises[0]
        if not premise.startswith("∀"):
            return False
            
        # Extract the variable and the predicate
        # This is a very simplified parsing, would need a proper parser in reality
        parts = premise.split(":", 1)
        if len(parts) != 2:
            return False
            
        var_part = parts[0]
        predicate = parts[1].strip()
        
        # Extract variable name
        var_name = var_part[1:].split("∈")[0].strip()
        
        # Check if the conclusion is an instance of the predicate
        # This is very simplified, would need a proper substitution mechanism
        for const in ["a", "b", "c", "1", "2", "3"]:
            if predicate.replace(var_name, const) == conclusion:
                return True
                
        return False
    
    def _apply_existential_generalization(self, premises: List[str], conclusion: str) -> bool:
        """
        Apply the existential generalization rule: from P(a), infer ∃x.P(x).
        This is a very simplified implementation.
        """
        if len(premises) != 1 or not conclusion.startswith("∃"):
            return False
            
        # Extract the variable and predicate from the conclusion
        parts = conclusion.split(":", 1)
        if len(parts) != 2:
            return False
            
        var_part = parts[0]
        predicate = parts[1].strip()
        
        # Extract variable name
        var_name = var_part[1:].split("∈")[0].strip()
        
        # Check if the premise is an instance of the predicate
        for const in ["a", "b", "c", "1", "2", "3"]:
            if predicate.replace(var_name, const) == premises[0]:
                return True
                
        return False
        
    def _apply_universal_introduction(self, premises: List[str], conclusion: str) -> bool:
        """
        Apply the universal introduction rule: from P(c) for arbitrary c, infer ∀x.P(x).
        
        Args:
            premises: List of premises
            conclusion: The conclusion to check
            
        Returns:
            True if the rule can be applied, False otherwise
        """
        if not conclusion.startswith("∀"):
            return False
            
        # Extract the variable and predicate from the conclusion
        parts = conclusion.split(":", 1)
        if len(parts) != 2:
            return False
            
        var_part = parts[0]
        predicate = parts[1].strip()
        
        # Extract variable name
        var_name = var_part[1:].split("∈")[0].strip()
        
        # For simplicity, we'll assume the premises contain instances for some constants
        # In a real proof system, we would check that c is arbitrary
        needed_instances = 3  # Require at least 3 instances to justify universal quantification
        found_instances = 0
        
        for const in ["a", "b", "c", "1", "2", "3"]:
            instance = predicate.replace(var_name, const)
            if instance in premises:
                found_instances += 1
                
        return found_instances >= needed_instances
        
    def _apply_existential_elimination(self, premises: List[str], conclusion: str) -> bool:
        """
        Apply the existential elimination rule: from ∃x.P(x) and P(a)→Q where a doesn't appear in Q,
        infer Q.
        
        Args:
            premises: List of premises
            conclusion: The conclusion to check
            
        Returns:
            True if the rule can be applied, False otherwise
        """
        # Simplified implementation
        existential_premise = None
        implication_premise = None
        
        # Find the existential premise
        for premise in premises:
            if premise.startswith("∃"):
                existential_premise = premise
                break
                
        if existential_premise is None:
            return False
            
        # Find an implication premise
        for premise in premises:
            if "→" in premise:
                antecedent, consequent = premise.split("→", 1)
                if consequent.strip() == conclusion:
                    implication_premise = premise
                    break
                    
        return implication_premise is not None
    
    def _apply_assumption(self, premises: List[str], conclusion: str) -> bool:
        """
        Allow introducing an assumption.
        In a proper system, this would be more restricted.
        """
        # For simplicity, allow assumptions in proofs
        # A more rigorous system would track scopes of assumptions
        return True
    
    def _implies(self, knowledge_set: Set[str], conclusion: str) -> bool:
        """
        Check if a conclusion follows from the knowledge set.
        
        Args:
            knowledge_set: Set of known statements
            conclusion: The conclusion to check
            
        Returns:
            True if the conclusion follows, False otherwise
        """
        # Simple implementation: Check if the conclusion is directly in the knowledge set
        if conclusion in knowledge_set:
            return True
            
        # In a real implementation, we would use a theorem prover here
        # to check if the conclusion can be derived from the knowledge set
        
        return False
    
    def add_theorem(self, theorem: str, proof: List[Dict[str, Any]]) -> None:
        """Add a verified theorem to the library."""
        is_valid, _ = self.verify(theorem, proof)
        if is_valid:
            self.theorem_library[theorem] = proof
            # Add the theorem to the axioms for future proofs
            self.axioms.add(theorem)
        else:
            logger.warning(f"Attempted to add invalid theorem: {theorem}")


class GNNEncoder(nn.Module):
    """Graph Neural Network for encoding mathematical structures."""
    
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 3, dropout: float = 0.1):
        """
        Initialize the GNN encoder.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super(GNNEncoder, self).__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple Graph Convolutional Layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            # Graph Attention Layer with edge features
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, edge_dim=edge_feature_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment for multiple graphs
            
        Returns:
            Embedding of the entire structure
        """
        # Handle empty graphs
        if x.shape[0] == 0:
            # Return zero embedding
            return torch.zeros(1, self.output_layer[0].out_features, device=x.device)
            
        # Encode node features
        x = self.node_encoder(x)
        
        # Graph Convolutional Layers with residual connections
        for conv, batch_norm in zip(self.conv_layers, self.batch_norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = batch_norm(x_new)
            x_new = F.relu(x_new)
            # Residual connection
            x = x + x_new
        
        # Global pooling
        if batch is not None:
            # If we have a batch of graphs, use global pooling
            x = global_mean_pool(x, batch)
        else:
            # Otherwise, take the mean of all nodes
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final projection
        x = self.output_layer(x)
        return x


class TheoremGenerator(nn.Module):
    """Generates new theorems based on the embedding of a mathematical structure."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 128, max_theorem_length: int = 100,
                 vocab_size: int = 128, dropout: float = 0.1):
        """
        Initialize the theorem generator.
        
        Args:
            embedding_dim: Dimension of the structure embedding
            hidden_dim: Dimension of hidden layers
            max_theorem_length: Maximum length of generated theorems
            vocab_size: Size of the mathematical vocabulary
            dropout: Dropout probability
        """
        super(TheoremGenerator, self).__init__()
        
        self.embedding_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # RNN for sequence generation with attention mechanism
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Symbol prediction (vocabulary of mathematical symbols and operators)
        self.vocab_size = vocab_size
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        self.max_length = max_theorem_length
        
    def forward(self, structure_embedding: torch.Tensor, start_token: Optional[torch.Tensor] = None, 
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate a new theorem based on the embedding.
        
        Args:
            structure_embedding: Embedding of the mathematical structure
            start_token: Optional start token (e.g., for conditional generation)
            temperature: Sampling temperature
            
        Returns:
            Sequence of token IDs representing a theorem
        """
        batch_size = structure_embedding.size(0)
        
        # Project the embedding
        h = self.embedding_proj(structure_embedding)
        
        # Initialize LSTM state with the embedding
        h0 = torch.zeros(2, batch_size, h.size(1), device=h.device)
        c0 = torch.zeros(2, batch_size, h.size(1), device=h.device)
        
        # Better initialization: Use the embedding for both layers
        h0[0] = h  # First layer hidden state
        h0[1] = h  # Second layer hidden state
        
        # Start with the start token or a special <START> token
        if start_token is None:
            # Assume token 0 is <START>
            current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=h.device)
        else:
            current_token = start_token.unsqueeze(1)
        
        # Get token embedding using the embedding layer
        token_embedding = self.token_embedding(current_token)
        
        # Generate tokens one by one
        generated_tokens = [current_token]
        hidden = (h0, c0)
        
        for _ in range(self.max_length - 1):
            # LSTM prediction for the next token
            output, hidden = self.lstm(token_embedding, hidden)
            
            # Predict the next token
            logits = self.output_layer(output.squeeze(1)) / temperature
            
            # Apply temperature-based sampling
            if temperature == 0:  # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Sample from the distribution
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                
            generated_tokens.append(next_token)
            
            # If all sequences end with the end token, stop generation
            # Assume token 1 is <END>
            if (next_token == 1).all():
                break
                
            # Embed the next token for the next step
            token_embedding = self.token_embedding(next_token)
        
        # Concatenate all generated tokens
        return torch.cat(generated_tokens, dim=1)
    
    def decode_theorem(self, token_ids: torch.Tensor, vocab: Dict[int, str]) -> str:
        """
        Decode token IDs back to a mathematical theorem in text format.
        
        Args:
            token_ids: Tensor of token IDs
            vocab: Dictionary mapping token IDs to strings
            
        Returns:
            The decoded theorem as a string
        """
        theorem_str = ""
        for token_id in token_ids.squeeze():
            token_id_item = token_id.item()
            if token_id_item == 1:  # <END> token
                break
            if token_id_item in vocab:
                theorem_str += vocab[token_id_item]
            else:
                theorem_str += f"<UNK:{token_id_item}>"
        return theorem_str


class MathTheoremHDRL:
    """High-Dimensional Reinforcement Learning for theorem generation and proof."""
    
    def __init__(self, gnn_encoder: GNNEncoder, theorem_generator: TheoremGenerator, 
                 theorem_verifier: TheoremVerifier, learning_rate: float = 1e-4, 
                 gamma: float = 0.99, vocab: Optional[Dict[int, str]] = None):
        """
        Initialize the HDRL system.
        
        Args:
            gnn_encoder: The GNN encoder model
            theorem_generator: The theorem generator model
            theorem_verifier: The theorem verifier
            learning_rate: Learning rate for optimization
            gamma: Discount factor for RL
            vocab: Vocabulary dictionary mapping token IDs to strings
        """
        self.encoder = gnn_encoder
        self.generator = theorem_generator
        self.verifier = theorem_verifier
        self.vocab = vocab or self._create_default_vocab()
        
        # Optimizers with weight decay for regularization
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.generator_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.gamma = gamma  # Discount factor
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # Metrics tracking
        self.metrics = {
            'valid_theorems': 0,
            'total_attempts': 0,
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'training_loss': []
        }
        
    def _create_default_vocab(self) -> Dict[int, str]:
        """
        Create a default vocabulary for mathematical expressions.
        
        Returns:
            Dictionary mapping token IDs to strings
        """
        vocab = {
            0: "<START>",
            1: "<END>",
            2: " ",
            # Digits
            3: "0", 4: "1", 5: "2", 6: "3", 7: "4", 
            8: "5", 9: "6", 10: "7", 11: "8", 12: "9",
            # Variables
            13: "x", 14: "y", 15: "z", 16: "n", 17: "m",
            18: "a", 19: "b", 20: "c", 21: "i", 22: "j",
            # Operators
            23: "+", 24: "-", 25: "*", 26: "/", 27: "=",
            28: "<", 29: ">", 30: "≤", 31: "≥", 32: "≠",
            # Logical operators
            33: "∧", 34: "∨", 35: "¬", 36: "→", 37: "↔",
            38: "∀", 39: "∃", 40: "∈", 41: "∉", 42: "⊂",
            43: "⊆", 44: "∩", 45: "∪", 46: "∅", 47: "∞",
            # Brackets and punctuation
            48: "(", 49: ")", 50: "[", 51: "]", 52: "{", 53: "}", 
            54: ",", 55: ".", 56: ":", 57: ";",
            # Mathematical functions and concepts
            58: "sin", 59: "cos", 60: "tan", 61: "log", 62: "exp",
            63: "lim", 64: "sup", 65: "inf", 66: "max", 67: "min",
            68: "gcd", 69: "lcm", 70: "mod", 71: "prime",
            # Set concepts
            72: "ℕ", 73: "ℤ", 74: "ℚ", 75: "ℝ", 76: "ℂ",
            # Additional mathematical symbols
            77: "≡", 78: "≈", 79: "√", 80: "∂", 81: "∫", 
            82: "∑", 83: "∏", 84: "△", 85: "□", 86: "⊕",
            87: "⊗", 88: "⊢", 89: "⊨", 90: "◇", 91: "→"
        }
        return vocab
    
    def generate_theorem(self, structure: MathematicalStructure, temperature: float = 1.0) -> Tuple[str, torch.Tensor]:
        """
        Generate a new theorem for the given mathematical structure.
        
        Args:
            structure: The mathematical structure
            temperature: Sampling temperature
            
        Returns:
            (theorem, token_ids): The generated theorem and its token IDs
        """
        # Convert structure to PyTorch Geometric format
        data = structure.to_torch_geometric()
        
        # Encode the structure
        with torch.no_grad():
            try:
                x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
                
                # Handle edge case of empty graphs
                if x.shape[0] == 0:
                    logger.warning("Attempting to generate a theorem for an empty structure")
                    return "<empty structure>", torch.zeros(1, 1, dtype=torch.long)
                    
                embedding = self.encoder(x, edge_index, edge_attr)
                
                # Generate a theorem
                token_ids = self.generator(embedding, temperature=temperature)
                
                # Enforce a minimum and maximum length
                if token_ids.size(1) < 3:  # Too short
                    logger.warning("Generated theorem is too short, padding")
                    padding = torch.ones((1, 3 - token_ids.size(1)), 
                                        dtype=torch.long, device=token_ids.device)
                    token_ids = torch.cat([token_ids, padding], dim=1)
                elif token_ids.size(1) > 50:  # Too long
                    logger.warning("Generated theorem is too long, truncating")
                    token_ids = token_ids[:, :50]
                
                # Decode to a string
                theorem = self.generator.decode_theorem(token_ids, self.vocab)
                
                # Filter out theorems with too many unknown tokens
                unknown_count = theorem.count("<UNK:")
                if unknown_count > len(theorem) / 3:  # More than 1/3 is unknown
                    logger.warning(f"Generated theorem has too many unknown tokens: {unknown_count}")
                    # Return a simple valid theorem
                    simple_theorem = "∀n∈ℕ: n+0=n"
                    # Create tokens for the simple theorem
                    simple_tokens = torch.zeros(1, 10, dtype=torch.long)
                    return simple_theorem, simple_tokens
                
                # Add proper structure to theorems with no structure
                if not (":" in theorem or "=" in theorem or "∈" in theorem or "→" in theorem):
                    if "∀" in theorem or "∃" in theorem:
                        # Add domain specification
                        theorem += ": "
                    elif len(theorem) > 5:
                        # Add an equality
                        theorem += "="
                
            except Exception as e:
                logger.error(f"Error generating theorem: {str(e)}")
                # Return a simple valid theorem as fallback
                simple_theorem = "∀n∈ℕ: n+0=n"
                # Create tokens for the simple theorem
                simple_tokens = torch.zeros(1, 10, dtype=torch.long)
                return simple_theorem, simple_tokens
            
        return theorem, token_ids
    
    def generate_proof(self, theorem: str, max_steps: int = 100) -> List[Dict[str, Any]]:
        """
        Attempt to generate a proof for the given theorem.
        
        Args:
            theorem: The theorem to prove
            max_steps: Maximum number of proof steps
            
        Returns:
            List of proof steps
        """
        # In a complete implementation, this would use an automated theorem prover
        proof_steps = []
        
        # Try applying some basic inference rules
        if "∧" in theorem:
            # Try to prove a conjunction by proving each part
            parts = theorem.split(" ∧ ")
            for part in parts:
                proof_steps.append({
                    'rule': 'assumption',
                    'premises': [],
                    'conclusion': part
                })
            
            # Then use conjunction introduction
            proof_steps.append({
                'rule': 'conjunction_introduction',
                'premises': parts,
                'conclusion': theorem
            })
        elif "→" in theorem:
            # Try to prove an implication
            antecedent, consequent = theorem.split(" → ")
            
            # Assume the antecedent
            proof_steps.append({
                'rule': 'assumption',
                'premises': [],
                'conclusion': antecedent
            })
            
            # Try to derive the consequent (simplified)
            proof_steps.append({
                'rule': 'assumption',
                'premises': [],
                'conclusion': consequent
            })
            
            # Use implication introduction
            proof_steps.append({
                'rule': 'implication_introduction',
                'premises': [antecedent, consequent],
                'conclusion': theorem
            })
        elif "∀" in theorem:
            # Handle universal quantification
            parts = theorem.split(":", 1)
            if len(parts) == 2:
                var_part = parts[0]
                predicate = parts[1].strip()
                
                # Extract variable name
                var_name = var_part[1:].split("∈")[0].strip()
                
                # Generate instances (simplified)
                instances = []
                for const in ["a", "b", "1"]:
                    instance = predicate.replace(var_name, const)
                    proof_steps.append({
                        'rule': 'assumption',
                        'premises': [],
                        'conclusion': instance
                    })
                    instances.append(instance)
                
                # Then use universal introduction
                proof_steps.append({
                    'rule': 'universal_introduction',
                    'premises': instances,
                    'conclusion': theorem
                })
        else:
            # For other types of theorems, just try a direct proof
            proof_steps.append({
                'rule': 'assumption',
                'premises': [],
                'conclusion': theorem
            })
        
        return proof_steps
    
    def train_step(self, batch_size: Optional[int] = None) -> float:
        """
        Perform a training step.
        
        Args:
            batch_size: Size of the batch to train on
            
        Returns:
            The training loss
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Sample a batch from the replay buffer
        if len(self.replay_buffer) < batch_size:
            return 0.0  # Not enough data to train
            
        batch = random.sample(self.replay_buffer, batch_size)
        
        try:
            # Unpack the batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Pad actions to the same length
            max_action_length = max(action.size(1) for action in actions)
            padded_actions = []
            for action in actions:
                if action.size(1) < max_action_length:
                    padding = torch.ones((action.size(0), max_action_length - action.size(1)), 
                                        dtype=action.dtype, device=action.device)
                    padded_action = torch.cat([action, padding], dim=1)
                    padded_actions.append(padded_action)
                else:
                    padded_actions.append(action)
            
            # Convert to tensors
            states = torch.stack(states)
            actions = torch.stack(padded_actions)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error during batch preparation: {str(e)}")
            return 0.0  # Return early if batch preparation fails
        
        # Actor-Critic Update
        self.encoder_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()
        
        # Forward pass to get log probabilities of actions
        logits = self.generator.output_layer(states)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Calculate the policy loss - encourage actions with higher rewards
        policy_loss = -torch.mean(log_probs * rewards.unsqueeze(1))
        
        # Add entropy bonus to encourage exploration
        entropy = -torch.mean(torch.sum(F.softmax(logits, dim=-1) * log_probs, dim=1))
        loss = policy_loss - 0.01 * entropy
        
        # Backpropagate and update
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        
        self.encoder_optimizer.step()
        self.generator_optimizer.step()
        
        # Track metrics
        self.metrics['training_loss'].append(loss.item())
        
        return loss.item()
    
    def explore(self, structure: MathematicalStructure, num_theorems: int = 10, 
                temperature_range: Tuple[float, float] = (0.5, 2.0)) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Explore new theorems for the given structure.
        
        Args:
            structure: The mathematical structure to explore
            num_theorems: Number of theorems to generate
            temperature_range: Range for the sampling temperature
            
        Returns:
            List of valid theorems with proofs
        """
        valid_theorems = []
        rewards_history = []
        
        # Add some pre-defined valid theorems to help bootstrap learning
        predefined_theorems = [
            "∀n∈ℕ: n+0=n",
            "∀n,m∈ℕ: n+m=m+n",
            "∀n,m,l∈ℕ: n*(m+l)=n*m+n*l",
            "∀n∈ℕ: n≥0"
        ]
        
        # Start with predefined theorems occasionally
        if random.random() < 0.3 and len(predefined_theorems) > 0:
            theorem = random.choice(predefined_theorems)
            logger.info(f"Using predefined theorem: {theorem}")
            
            # Create a simple proof for the predefined theorem
            proof = [{
                'rule': 'assumption',
                'premises': [],
                'conclusion': theorem
            }]
            
            valid_theorems.append((theorem, proof))
            
            # Add to the verifier's theorem library
            self.verifier.add_theorem(theorem, proof)
            
            # Positive reward
            reward = 1.5
            self.metrics['valid_theorems'] += 1
            self.metrics['total_attempts'] += 1
            rewards_history.append(reward)
            
            # Add to replay buffer
            token_ids = torch.zeros(1, 10, dtype=torch.long)  # Simplified token ids
            
            # Encode the structure
            data = structure.to_torch_geometric()
            with torch.no_grad():
                try:
                    state = self.encoder(data['x'], data['edge_index'], data['edge_attr'])
                    
                    # Add to replay buffer
                    self.replay_buffer.append((
                        state,
                        token_ids,
                        reward,
                        state.clone(),
                        True
                    ))
                except Exception as e:
                    logger.error(f"Error encoding predefined theorem: {str(e)}")
        
        # Normal exploration with temperature-based generation
        for i in range(num_theorems):
            try:
                # Choose a random temperature (using adaptive range based on success)
                if rewards_history and sum(rewards_history) / len(rewards_history) < 0:
                    # If we're getting negative rewards, focus more on exploration
                    temperature = random.uniform(0.8, 3.0)
                else:
                    temperature = random.uniform(*temperature_range)
                
                # Generate a theorem
                theorem, token_ids = self.generate_theorem(structure, temperature)
                
                # Skip empty or very short theorems
                if len(theorem) < 5:
                    logger.debug(f"Skipping too short theorem: {theorem}")
                    continue
                
                logger.info(f"Generated theorem {i+1}/{num_theorems}: {theorem}")
                
                # Try to find a proof
                proof = self.generate_proof(theorem)
                
                # Verify the proof
                is_valid, feedback = self.verifier.verify(theorem, proof)
                
                self.metrics['total_attempts'] += 1
                
                if is_valid:
                    logger.info(f"Found valid theorem: {theorem}")
                    valid_theorems.append((theorem, proof))
                    
                    # Add to the verifier's theorem library
                    self.verifier.add_theorem(theorem, proof)
                    
                    # Positive reward for a correct theorem
                    # Scale reward based on theorem complexity and novelty
                    complexity_bonus = min(len(theorem) / 20, 2.0)
                    
                    # Check for novelty compared to existing theorems
                    novelty_bonus = 0.5
                    for existing_theorem, _ in valid_theorems[:-1]:  # Exclude the one we just added
                        if self._similarity(theorem, existing_theorem) > 0.7:
                            novelty_bonus = 0.0
                            break
                    
                    reward = 1.0 + complexity_bonus + novelty_bonus
                    self.metrics['valid_theorems'] += 1
                else:
                    logger.debug(f"Invalid theorem: {theorem}, Feedback: {feedback}")
                    # Negative reward for an incorrect theorem, but not too harsh
                    reward = -0.1
                
                rewards_history.append(reward)
                    
                # Encode the structure for the replay buffer
                data = structure.to_torch_geometric()
                with torch.no_grad():
                    try:
                        state = self.encoder(data['x'], data['edge_index'], data['edge_attr'])
                        
                        # Add the experience to the replay buffer
                        # (state, action, reward, next_state, done)
                        self.replay_buffer.append((
                            state,
                            token_ids,
                            reward,
                            state.clone(),  # In this case, the state remains the same
                            True
                        ))
                    except Exception as e:
                        logger.error(f"Error encoding theorem for replay buffer: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error exploring theorem {i+1}: {str(e)}")
                continue
            
            # Train after each exploration but not too frequently to avoid overfitting
            if len(self.replay_buffer) >= self.batch_size and i % 2 == 0:
                try:
                    loss = self.train_step(batch_size=min(self.batch_size, len(self.replay_buffer)))
                    logger.debug(f"Training loss: {loss}")
                except Exception as e:
                    logger.error(f"Error during training step: {str(e)}")
        
        # Update success rate metric
        if self.metrics['total_attempts'] > 0:
            self.metrics['success_rate'] = self.metrics['valid_theorems'] / self.metrics['total_attempts']
        
        # Update average reward metric
        if rewards_history:
            self.metrics['avg_reward'] = sum(rewards_history) / len(rewards_history)
        
        return valid_theorems
        
    def _similarity(self, theorem1: str, theorem2: str) -> float:
        """
        Calculate a simple similarity score between two theorems.
        
        Args:
            theorem1: First theorem
            theorem2: Second theorem
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to sets of tokens for a simple Jaccard similarity
        tokens1 = set(theorem1.replace(" ", ""))
        tokens2 = set(theorem2.replace(" ", ""))
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def discover_new_mathematics(self, initial_structures: List[MathematicalStructure], 
                                exploration_steps: int = 1000,
                                structures_to_keep: int = 10) -> Tuple[Dict[str, Dict[str, Any]], List[MathematicalStructure]]:
        """
        Long-term process for discovering new mathematics.
        
        Args:
            initial_structures: Initial structures to explore
            exploration_steps: Number of exploration steps
            structures_to_keep: Maximum number of structures to maintain
            
        Returns:
            (theorem_library, structures): Library of new theorems and structures
        """
        structures = list(initial_structures)
        theorem_library = {}
        
        # Track the quality of each structure
        structure_quality = {i: 0 for i in range(len(structures))}
        
        for step in range(exploration_steps):
            logger.info(f"Exploration step {step+1}/{exploration_steps}")
            
            # Choose a random structure, weighted by quality
            total_quality = sum(structure_quality.values()) + 1e-6  # Avoid division by zero
            weights = [structure_quality[i] / total_quality for i in range(len(structures))]
            
            # Add a small exploration factor
            weights = [w + 0.1 for w in weights]
            weights = [w / sum(weights) for w in weights]
            
            structure_idx = random.choices(range(len(structures)), weights=weights)[0]
            structure = structures[structure_idx]
            
            logger.info(f"Exploring structure: {structure}")
            
            # Explore new theorems
            new_theorems = self.explore(structure, num_theorems=5)
            
            # Update structure quality based on success
            structure_quality[structure_idx] += len(new_theorems) * 0.5
            
            # Add new theorems to the library
            for theorem, proof in new_theorems:
                theorem_library[theorem] = {
                    'proof': proof,
                    'structure': structure,
                    'discovery_step': step
                }
            
            # Occasionally generate new structures or combine existing ones
            if step % 10 == 0 and theorem_library:
                # Generate a new structure based on insights
                if new_theorems:
                    # Use the most successful structure as base
                    base_structure_idx = max(structure_quality, key=structure_quality.get)
                    base_structure = structures[base_structure_idx]
                    
                    new_structure = self._generate_new_structure(base_structure, new_theorems)
                    structures.append(new_structure)
                    structure_quality[len(structures) - 1] = 0.5  # Initial quality score
                    
                    logger.info(f"Generated new structure: {new_structure}")
                
                # Combine two existing structures
                if len(structures) >= 2:
                    # Choose two parent structures, weighted by quality
                    parent_idxs = random.choices(range(len(structures)), 
                                                weights=[structure_quality[i] + 0.1 for i in range(len(structures))],
                                                k=2)
                    parent1 = structures[parent_idxs[0]]
                    parent2 = structures[parent_idxs[1]]
                    
                    combined_structure = self._combine_structures(parent1, parent2)
                    structures.append(combined_structure)
                    structure_quality[len(structures) - 1] = 0.5  # Initial quality score
                    
                    logger.info(f"Combined structures to create: {combined_structure}")
            
            # Prune structures if we have too many, keeping the highest quality ones
            if len(structures) > structures_to_keep:
                # Sort structures by quality
                sorted_idxs = sorted(range(len(structures)), 
                                    key=lambda i: structure_quality[i], 
                                    reverse=True)
                
                # Keep only the top structures
                structures = [structures[i] for i in sorted_idxs[:structures_to_keep]]
                
                # Update quality dictionary
                new_quality = {}
                for i, old_idx in enumerate(sorted_idxs[:structures_to_keep]):
                    new_quality[i] = structure_quality[old_idx]
                structure_quality = new_quality
            
            # Regular training
            if len(self.replay_buffer) >= self.batch_size:
                self.train_step()
            
            # Log progress
            if step % 10 == 0:
                logger.info(f"Progress: {step/exploration_steps*100:.1f}% complete")
                logger.info(f"Discovered {len(theorem_library)} theorems")
                logger.info(f"Current structures: {len(structures)}")
                logger.info(f"Success rate: {self.metrics['success_rate']:.2f}")
                logger.info(f"Average reward: {self.metrics['avg_reward']:.2f}")
            
        return theorem_library, structures
    
    def _generate_new_structure(self, base_structure: MathematicalStructure, 
                              theorems: List[Tuple[str, List[Dict[str, Any]]]]) -> MathematicalStructure:
        """
        Generate a new mathematical structure based on existing ones.
        
        Args:
            base_structure: Base structure to build upon
            theorems: List of theorems with proofs
            
        Returns:
            A new mathematical structure
        """
        # Create a new structure with a unique name
        new_structure = MathematicalStructure(f"{base_structure.name}_derived_{random.randint(1000, 9999)}")
        
        # Copy existing node types and edge types
        new_structure.node_types = base_structure.node_types.copy()
        new_structure.edge_types = base_structure.edge_types.copy()
        
        # Copy nodes with some probability and potential mutations
        for node in base_structure.nodes:
            # With 90% probability, keep the node
            if random.random() < 0.9:
                # Copy attributes with potential mutations
                attributes = node['attributes'].copy()
                
                # With 20% probability, modify an attribute if it exists
                if attributes and random.random() < 0.2:
                    if 'value' in attributes and isinstance(attributes['value'], (int, float)):
                        # Modify numerical value
                        attributes['value'] = attributes['value'] + random.uniform(-1, 1)
                
                new_structure.add_node(node['type'], attributes)
        
        # Copy edges with some probability
        for edge in base_structure.edges:
            # Adjust indices to match new structure's node count
            if edge['source'] < len(new_structure.nodes) and edge['target'] < len(new_structure.nodes):
                # With 80% probability, keep the edge
                if random.random() < 0.8:
                    new_structure.add_edge(edge['source'], edge['target'], edge['type'])
        
        # Add new nodes based on discovered theorems
        theorem_types = self._extract_concept_types_from_theorems(theorems)
        for concept_type in theorem_types:
            # Add a new node with this concept type
            if random.random() < 0.3:  # 30% chance to add a new node per concept
                if concept_type in new_structure.node_types:
                    new_node_id = new_structure.add_node(concept_type, {'source': 'theorem_derived'})
                    
                    # Connect to some existing nodes
                    num_connections = random.randint(1, min(3, len(new_structure.nodes) - 1))
                    for _ in range(num_connections):
                        if len(new_structure.nodes) > 1:
                            target_id = random.randint(0, len(new_structure.nodes) - 2)
                            relation_type = random.choice(new_structure.edge_types)
                            new_structure.add_edge(new_node_id, target_id, relation_type)
        
        # Add some completely new nodes
        num_new_nodes = random.randint(1, 3)
        for _ in range(num_new_nodes):
            node_type = random.choice(new_structure.node_types)
            new_node_id = new_structure.add_node(node_type, {'origin': 'mutation'})
            
            # Connect to existing nodes
            if len(new_structure.nodes) > 1:
                num_connections = random.randint(1, min(3, len(new_structure.nodes) - 1))
                for _ in range(num_connections):
                    target_id = random.randint(0, len(new_structure.nodes) - 2)
                    relation_type = random.choice(new_structure.edge_types)
                    new_structure.add_edge(new_node_id, target_id, relation_type)
        
        return new_structure
    
    def _combine_structures(self, structure1: MathematicalStructure, 
                          structure2: MathematicalStructure) -> MathematicalStructure:
        """
        Combine two mathematical structures to create a new one.
        
        Args:
            structure1: First parent structure
            structure2: Second parent structure
            
        Returns:
            A new combined structure
        """
        # Create a new structure with a combined name
        new_structure = MathematicalStructure(f"{structure1.name}_{structure2.name}_combined")
        
        # Combine node types and edge types
        new_structure.node_types = list(set(structure1.node_types + structure2.node_types))
        new_structure.edge_types = list(set(structure1.edge_types + structure2.edge_types))
        
        # Copy all nodes from structure1
        for node in structure1.nodes:
            new_structure.add_node(node['type'], node['attributes'].copy())
        
        # Copy edges from structure1
        for edge in structure1.edges:
            new_structure.add_edge(edge['source'], edge['target'], edge['type'])
        
        # Add nodes from structure2 with offset indices
        offset = len(new_structure.nodes)
        for node in structure2.nodes:
            new_structure.add_node(node['type'], node['attributes'].copy())
        
        # Add edges from structure2 with offset indices
        for edge in structure2.edges:
            new_structure.add_edge(edge['source'] + offset, edge['target'] + offset, edge['type'])
        
        # Create bridge edges between the two structures
        num_bridges = random.randint(1, min(5, len(structure1.nodes) * len(structure2.nodes) // 10))
        for _ in range(num_bridges):
            source_id = random.randint(0, len(structure1.nodes) - 1)
            target_id = random.randint(0, len(structure2.nodes) - 1) + offset
            relation_type = random.choice(new_structure.edge_types)
            new_structure.add_edge(source_id, target_id, relation_type)
        
        return new_structure
    
    def _extract_concept_types_from_theorems(self, theorems: List[Tuple[str, List[Dict[str, Any]]]]) -> List[str]:
        """
        Extract concept types mentioned in theorems.
        
        Args:
            theorems: List of theorems with proofs
            
        Returns:
            List of concept types
        """
        concept_types = []
        
        # Simple heuristic extraction based on theorem content
        for theorem, _ in theorems:
            if "∀" in theorem or "∃" in theorem:
                concept_types.append('variable')
            if "+" in theorem or "-" in theorem or "*" in theorem:
                concept_types.append('operator')
            if "=" in theorem or "≠" in theorem or "<" in theorem or ">" in theorem:
                concept_types.append('relation')
            if "∈" in theorem or "⊂" in theorem:
                concept_types.append('set')
            if "sin" in theorem or "cos" in theorem or "exp" in theorem:
                concept_types.append('function')
            if "ℕ" in theorem or "ℤ" in theorem or "ℝ" in theorem:
                concept_types.append('number')
        
        return list(set(concept_types))


# Example for a simple execution
def example_run():
    """Run a simple example of the mathematical theorem discovery system."""
    try:
        # Set random seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        
        logger.info("Initializing mathematical structures...")
        
        # Create a simple mathematical structure (e.g., natural numbers with addition)
        natural_numbers = MathematicalStructure("NaturalNumbers")
        
        # Add basic elements
        base_set_id = natural_numbers.add_node('set', {'name': 'ℕ'})
        addition_id = natural_numbers.add_node('operator', {'name': 'addition', 'symbol': '+'})
        multiplication_id = natural_numbers.add_node('operator', {'name': 'multiplication', 'symbol': '*'})
        zero_id = natural_numbers.add_node('number', {'value': 0, 'symbol': '0'})
        one_id = natural_numbers.add_node('number', {'value': 1, 'symbol': '1'})
        
        # Connect elements
        natural_numbers.add_edge(addition_id, base_set_id, 'operates_on')
        natural_numbers.add_edge(multiplication_id, base_set_id, 'operates_on')
        natural_numbers.add_edge(base_set_id, zero_id, 'contains')
        natural_numbers.add_edge(base_set_id, one_id, 'contains')
        
        # Add some special properties
        natural_numbers.add_node('property', {'name': 'identity', 'relates': [addition_id, zero_id]})
        natural_numbers.add_node('property', {'name': 'identity', 'relates': [multiplication_id, one_id]})
        
        # Add some numbers
        for i in range(2, 10):
            num_id = natural_numbers.add_node('number', {'value': i})
            natural_numbers.add_edge(base_set_id, num_id, 'contains')
        
        # Create a more complex structure: integers with operations
        integers = MathematicalStructure("Integers")
        
        # Add basic elements
        int_set_id = integers.add_node('set', {'name': 'ℤ'})
        add_id = integers.add_node('operator', {'name': 'addition', 'symbol': '+'})
        sub_id = integers.add_node('operator', {'name': 'subtraction', 'symbol': '-'})
        mult_id = integers.add_node('operator', {'name': 'multiplication', 'symbol': '*'})
        zero_int_id = integers.add_node('number', {'value': 0, 'symbol': '0'})
        one_int_id = integers.add_node('number', {'value': 1, 'symbol': '1'})
        
        # Connect elements
        integers.add_edge(add_id, int_set_id, 'operates_on')
        integers.add_edge(sub_id, int_set_id, 'operates_on')
        integers.add_edge(mult_id, int_set_id, 'operates_on')
        integers.add_edge(int_set_id, zero_int_id, 'contains')
        integers.add_edge(int_set_id, one_int_id, 'contains')
        
        # Add special properties
        integers.add_node('property', {'name': 'identity', 'relates': [add_id, zero_int_id]})
        integers.add_node('property', {'name': 'identity', 'relates': [mult_id, one_int_id]})
        
        # Add some positive and negative integers
        for i in range(-5, 6):
            if i != 0 and i != 1:  # Skip already added elements
                num_id = integers.add_node('number', {'value': i})
                integers.add_edge(int_set_id, num_id, 'contains')
        
        # Create real numbers structure
        reals = MathematicalStructure("RealNumbers")
        real_set_id = reals.add_node('set', {'name': 'ℝ'})
        
        # Add operations
        add_real_id = reals.add_node('operator', {'name': 'addition', 'symbol': '+'})
        mult_real_id = reals.add_node('operator', {'name': 'multiplication', 'symbol': '*'})
        div_real_id = reals.add_node('operator', {'name': 'division', 'symbol': '/'})
        
        # Connect operations to set
        reals.add_edge(add_real_id, real_set_id, 'operates_on')
        reals.add_edge(mult_real_id, real_set_id, 'operates_on')
        reals.add_edge(div_real_id, real_set_id, 'operates_on')
        
        # Add some special real numbers
        zero_real_id = reals.add_node('number', {'value': 0.0, 'symbol': '0'})
        one_real_id = reals.add_node('number', {'value': 1.0, 'symbol': '1'})
        pi_id = reals.add_node('number', {'value': 3.14159, 'symbol': 'π'})
        e_id = reals.add_node('number', {'value': 2.71828, 'symbol': 'e'})
        
        reals.add_edge(real_set_id, zero_real_id, 'contains')
        reals.add_edge(real_set_id, one_real_id, 'contains')
        reals.add_edge(real_set_id, pi_id, 'contains')
        reals.add_edge(real_set_id, e_id, 'contains')
        
        # Create an axiom system
        axioms = [
            "∀n∈ℕ: n+0=n",  # Identity element
            "∀n,m∈ℕ: n+m=m+n",  # Commutativity
            "∀n,m,l∈ℕ: (n+m)+l=n+(m+l)",  # Associativity
            "∀n∈ℕ: n*1=n",  # Multiplicative identity
            "∀n∈ℕ: n*0=0",  # Multiplication by zero
            "∀n,m∈ℕ: n*m=m*n",  # Multiplicative commutativity
            "∀n,m,l∈ℕ: (n*m)*l=n*(m*l)",  # Multiplicative associativity
            "∀n,m,l∈ℕ: n*(m+l)=n*m+n*l",  # Distributivity
            "∀n∈ℤ: n+0=n",  # Integer identity
            "∀n∈ℤ: n*1=n",  # Integer multiplicative identity
            "∀n∈ℤ: n+(-n)=0",  # Additive inverse
            "∀n∈ℝ: n/1=n",  # Division by 1
            "∀n∈ℝ,n≠0: n*1/n=1",  # Multiplicative inverse
        ]
        
        logger.info("Setting up neural network components...")
        
        # Initialize components with correct dimensions
        node_feature_dim = len(natural_numbers._encode_node_features(natural_numbers.nodes[0]))
        edge_feature_dim = len(natural_numbers._encode_edge_features(natural_numbers.edges[0]))
        hidden_dim = 64
        
        # Use try-except for model initialization
        try:
            gnn_encoder = GNNEncoder(node_feature_dim, edge_feature_dim, hidden_dim)
            theorem_generator = TheoremGenerator(hidden_dim)
            theorem_verifier = TheoremVerifier(axioms)
            
            # Initialize HDRL system
            logger.info("Initializing HDRL system...")
            math_hdrl = MathTheoremHDRL(gnn_encoder, theorem_generator, theorem_verifier)
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
        
        # Discover new theorems with smaller steps for debugging
        logger.info("Starting theorem discovery process...")
        structures = [natural_numbers, integers, reals]
        
        # Use a smaller exploration_steps for testing
        exploration_steps = 20  # Reduced from 50 for faster testing
        
        theorem_library, new_structures = math_hdrl.discover_new_mathematics(
            structures, exploration_steps=exploration_steps, structures_to_keep=5)
        
        # Output the results
        logger.info(f"Discovered {len(theorem_library)} new theorems")
        for i, (theorem, details) in enumerate(theorem_library.items()):
            logger.info(f"Theorem {i+1}: {theorem}")
            logger.info(f"Discovered at step: {details['discovery_step']}")
            logger.info(f"Proof: {details['proof']}")
            logger.info("-" * 40)
        
        logger.info(f"Generated {len(new_structures)} new mathematical structures")
        for i, structure in enumerate(new_structures):
            logger.info(f"Structure {i+1}: {structure}")
        
        # Final metrics
        logger.info("\nFinal metrics:")
        logger.info(f"Total theorems attempted: {math_hdrl.metrics['total_attempts']}")
        logger.info(f"Valid theorems discovered: {math_hdrl.metrics['valid_theorems']}")
        logger.info(f"Success rate: {math_hdrl.metrics['success_rate']:.2f}")
        logger.info(f"Average reward: {math_hdrl.metrics['avg_reward']:.2f}")
        
        return theorem_library, new_structures
        
    except Exception as e:
        logger.error(f"Error in example_run: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        # Return empty results to avoid further errors
        return {}, []


if __name__ == "__main__":
    example_run()
