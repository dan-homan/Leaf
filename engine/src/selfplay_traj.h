// Binary per-game trajectory format (.tdg) — Stage 1 actor/learner split.
//
// An actor (frozen internal self-play, selfplay.cpp --traj-out DIR) writes one
// file per completed game: TDTrajHeader followed by n_records TDTrajRecord.
// The learner (--learn-stream DIR) consumes them in arrival order and runs the
// exact online update (tdleaf_update_after_game) with ONE optimizer.
//
// Only search outputs and POV/gate metadata are stored; accumulators, active
// features, and stack indices are rebuilt by the learner from the positions
// (tdleaf_rebuild_record) — integer accumulator rebuilds are exact, so the
// learner reproduces the online gradients bit-for-bit when run with the same
// starting state, env, and game order.
//
// Native-endian, same-machine format: guarded by magic/version and the v10
// source-.nnue content hash (actor and learner must load the same base net —
// the epoch-refresh protocol copies the learner's .tdleaf.bin state, so the
// base-net hash stays equal across refreshes).
//
// Write protocol: write to <name>.tdg.tmp, fclose, rename() to <name>.tdg —
// the learner never sees partial files.

#ifndef SELFPLAY_TRAJ_H
#define SELFPLAY_TRAJ_H

static const uint32_t TDTRAJ_MAGIC   = 0x31474454;  // "TDG1"
static const uint32_t TDTRAJ_VERSION = 2;           // v2: dropped hybrid-target gate keys

struct TDTrajHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t nnue_content_hash;   // nnue_get_content_hash() of the actor's base net
    float    result_white_pov;    // 1.0 / 0.5 / 0.0
    int32_t  n_records;
    uint32_t gid;                 // (pid & 0xFFF) << 20 | seq — TSV dump convention
};

struct TDTrajRecord {
    position pos;                 // leaf position  (acc/ft/stack rebuilt from it)
    position root_pos;            // root position  (dumped to the root-row TSV corpus)
    int32_t  score_stm;           // leaf static eval, leaf-STM POV (cp)
    int32_t  score_root_stm;      // root search score, root-STM POV (cp)
    int32_t  root_static;         // root static eval, root-STM POV (cp)
    int32_t  game_ply;            // 1-based game-ply of the root position
    float    id_score_variance;   // cp²
    int8_t   id_depth;
    uint8_t  wtm;                 // leaf STM
    uint8_t  root_wtm;            // root STM (alternates under self-play)
    uint8_t  _pad;
};

#endif // SELFPLAY_TRAJ_H
