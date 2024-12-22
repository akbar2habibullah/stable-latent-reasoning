# Algorithm: Stable Latent Reasoning (SLR)

**Input:**

*   `x`: Input query (sequence of tokens)
*   `T`: Maximum number of iterations (hyperparameter)
*   `f_θ`: Pre-trained Large Language Model with parameters `θ`
*   `L_t`: Loop-wise LoRA modules for each iteration `t` (either pre-trained or to be trained)
*   `E`: Timestep embedding function (either pre-trained or to be trained)
*   `decode`: Decoding function to generate output from the final hidden state
*   `α`: Residual connection weight (hyperparameter, typically between 0 and 1)
*   `condition`: function to determine whether the loop will terminate
*   `threshold`: threshold for early termination

**Output:**

*   `y`: Output sequence (e.g., answer to the query)

**Procedure:**

1. **Initialization:**
    *   Tokenize the input query `x` into a sequence of token embeddings.
    *   Encode the input embeddings using the LLM's embedding layer to obtain the initial hidden state: `h_0 = f_θ(x)`.
    *   Set iteration counter `t = 0`.
    *   Set the initial output sequence `y = ""`

2. **Iterative Refinement:**
    *   **While** `t < T` **and** `condition(h_t) < threshold`:
        *   Increment iteration counter: `t = t + 1`.
        *   **Timestep Embedding:**
            *   Compute the timestep embedding: `e_t = E(t)`.
        *   **Hidden State Update:**
            *   Apply the loop-specific LoRA module (if using): `θ_t = θ + L_t`
            *   Compute the new hidden state using the LLM with potentially modified parameters: `h_t' = f_{θ_t}(h_{t-1}, e_t)`. This step typically involves concatenating or adding `e_t` to `h_{t-1}` before passing it to the model.
        *   **Loop Residual Connection:**
            *   Combine the new hidden state with the initial hidden state or previous hidden state using the residual connection weight:
            *  Option A (with `h_0`): `h_t = α * h_t' + (1 - α) * h_0`
            *  Option B (with `h_{t-1}`): `h_t = α * h_t' + (1 - α) * h_{t-1}`
            *  Option C (other combination methods): `h_t = combine(h_t', h_0, h_{t-1})` where `combine` is a more complex learned function.
3. **Decoding:**
    *   Generate the output sequence from the final hidden state: `y = decode(h_t)`.

4. **Return:** `y`

**Detailed Explanations:**

*   **Timestep Embedding (`e_t`)**: This embedding provides information about the current loop iteration. The `E` function can be a simple lookup table or a more complex neural network. It can optionally also encode a "noise level."
*   **Loop-wise LoRA (`L_t`)**: These are low-rank matrices that modify the LLM's parameters `θ` at each iteration. They allow for fine-grained adjustments to the model's behavior during each step of the reasoning process.
*   **Residual Connection (`α`)**: The weight `α` controls the strength of the residual connection. A value of `α = 1` means no residual connection, while `α = 0` means only the initial/previous hidden state is used. Typical values are between 0.5 and 0.9.
*   **Decoding (`decode`)**: This function can be a simple linear projection of the final hidden state to the vocabulary space, followed by a softmax to generate probabilities for each token. Alternatively, it could be a more complex, separately trained decoder network.
*   **Condition function**: This function will check the condition of the hidden state whether the model should stop the loop. For example, when the hidden state doesn't change much anymore. This function will return a value, and if the value reach the `threshold`, the loop will terminate.
*   **Threshold**: A value to determine the early termination. The model will keep loop until the maximum loop or the value from `condition` function reach this threshold. If we don't use early termination, we can set the value to infinity.

**Training Considerations (Not Part of the Inference Algorithm):**

*   **End-to-End Training:**  All parameters (`θ`, `L_t`, `E`, `decode`, potentially `α` or parameters of `combine`) are trained jointly using a suitable loss function (e.g., cross-entropy loss for predicting the correct output sequence).
*   **Fine-tuning:** Start with a pre-trained LLM (`θ`) and only train `L_t`, `E`, `decode`, and potentially refine `θ`.
*   **Distillation:** Use a teacher model (e.g., a model trained with CoT) to provide supervision for the hidden states at each iteration.

**Variations:**

*   **Different Residual Connections:** Explore alternative ways of combining hidden states in the residual connection (e.g., gated mechanisms, attention mechanisms).
*   **Hybrid Approaches:** Combine different stabilization techniques (e.g., use both timestep embeddings and LoRA).
*   **Dynamic Iteration Count:** Instead of a fixed `T`, develop mechanisms to dynamically determine the number of iterations based on the input query or the state of the reasoning process.
*   **No Early Termination:** If we don't use early termination, then we don't need to define `condition` function and `threshold` value. We simply set the loop to a fixed number.

This detailed algorithm provides a concrete implementation of Stable Latent Reasoning. The specific choices for each component (e.g., the type of timestep embedding, the LoRA rank, the residual connection method) will depend on the specific application and can be determined through experimentation and hyperparameter tuning.
