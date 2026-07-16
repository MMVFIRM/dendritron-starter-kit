DENDRITRON — SPLIT OPTICAL DIGITS BENCHMARK

Dataset:
- scikit-learn Optical Recognition of Handwritten Digits
- 1797 real handwritten digit images
- 64 pixel features per 8x8 image
- stratified 70/30 train-test split per seed

Protocol:
- five sequential tasks: ['0/1', '2/3', '4/5', '6/7', '8/9']
- class-incremental inference: no task identity is supplied
- final prediction is over all ten digit classes
- 10 random train-test splits
- 25 training epochs per task
- no replay in the main Dendritron / fixed branch / standard MLP comparison

Capacity:
- Dendritron: 10240 learned scalar center values
- Fixed branches: 10240 learned scalar center values
- Backprop MLP: 10210 learned scalar parameters
- Difference between Dendritron and MLP: 30 scalars

Dendritron learning:
- 16 class-local branches per digit
- label-local branch allocation
- winner-take-all branch competition
- only the winning branch center updates
- no autodiff, backpropagation, or global gradient
- no task identity during inference

Replay control:
- Backprop MLP + replay stores 40 examples per learned class
- maximum replay memory after all tasks: 400 images

Damage test:
- disables the 25% most-used Dendritron / fixed branches
- disables the 25% most important MLP hidden units
- repair uses 40 examples per class
- only damaged/replacement structures are updated during repair

Interpretive limitation:
This is a prototype / LVQ-style realization of a Dendritron, not yet the full
dendritic architecture with emergent branch growth, eligibility traces,
multimodal modulators, recurrent coalitions, or autonomous context discovery.
