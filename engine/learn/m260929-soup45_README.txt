m260929-soup45: weight average of two sibling legs from m260929-4e6g-final
  m260929-4.5e6g_final.tdleaf.bin  (500k games, the intended leg)
  m260929-5e6g_final.tdleaf.bin    (500k games, accidentally also from 4e6g)
merged with scripts/merge_tdleaf.py (count-weighted ~48/52; Adam v max, m mean,
t max), .nnue baked by the engine's --write-nnue (validated: baking 4.5e6g's
state reproduces its final .nnue; a self-merge bakes to the identical net).
Depth 8, 8000 games vs classic_eval, --srand 20261004:
  4.5e6g-final +88.9 +- 3.6   5e6g-final +97.0 +- 3.7   soup +100.7 +- 3.6
Next leg: --continue m260929-4.5e6g --state m260929-soup45_final.tdleaf.bin
