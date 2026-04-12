// nnue_embed.cpp — embed the default NNUE net into the binary via incbin.
// Only compiled when NNUE=1 and NNUE_EMBED=1.
// NNUE_NET_PATH must be defined as a string literal path to the .nnue file.

#if NNUE && NNUE_EMBED

#define INCBIN_SILENCE_BITCODE_WARNING
#include "incbin.h"

INCBIN(NnueNet, NNUE_NET_PATH);

#endif
