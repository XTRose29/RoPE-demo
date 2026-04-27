# RoPE Demo Rationale Draft

This demo is designed around one core misconception from transformers: attention
does not automatically know token order. The first panel uses the sentence "I
like black coffee" from lecture and shows the actual numeric position vectors
for raw index, binary, sinusoidal, learned lookup, and RoPE-style angles. This
makes the abstract phrase "positional encoding" concrete.

The second panel focuses on RoPE. Instead of adding another vector to the token,
RoPE rotates each 2D query/key feature pair by a position-dependent angle.
Students can shift the entire sentence "The dog chased the cat" by +3 positions
and see that the dog-cat distance stays fixed. The RoPE similarity also stays
fixed because the angle gap depends on `m - n`, which explains why RoPE
naturally represents relative distance.

The third panel trains a tiny attention scorer on the task "attend to the token
two positions back." It trains only on short sequences and tests on longer
sequences, controlled by sliders. Learned absolute embeddings gray out unseen
positions because there is no learned vector for them, while RoPE can still
compute `angle = position * theta`. This is not a full language model; it
deliberately isolates the positional subproblem inside a transformer.
