# isort:skip_file

try:
    from . import checkpoint
except ImportError:
    checkpoint = None

try:
    from . import decode
except ImportError:
    decode = None

try:
    from . import dist
except ImportError:
    dist = None

try:
    from . import env
except ImportError:
    env = None

try:
    from . import utils
except ImportError:
    utils = None

try:
    from .byte_utils import (
        byte_decode,
        byte_encode,
        smart_byte_decode,
    )
except ImportError:
    byte_decode = byte_encode = smart_byte_decode = None

try:
    from .checkpoint import (
        average_checkpoints,
        find_checkpoints,
        load_checkpoint,
        remove_checkpoints,
        save_checkpoint,
        save_checkpoint_with_global_batch_idx,
    )
except ImportError:
    average_checkpoints = find_checkpoints = load_checkpoint = None
    remove_checkpoints = save_checkpoint = save_checkpoint_with_global_batch_idx = None

try:
    from .context_graph import ContextGraph, ContextState
except ImportError:
    ContextGraph = ContextState = None

try:
    from .decode import (
        get_lattice,
        nbest_decoding,
        nbest_oracle,
        one_best_decoding,
        rescore_with_attention_decoder,
        rescore_with_n_best_list,
        rescore_with_whole_lattice,
    )
except ImportError:
    get_lattice = nbest_decoding = nbest_oracle = one_best_decoding = None
    rescore_with_attention_decoder = rescore_with_n_best_list = rescore_with_whole_lattice = None

try:
    from .dist import (
        cleanup_dist,
        setup_dist,
    )
except ImportError:
    cleanup_dist = setup_dist = None

try:
    from .env import (
        get_env_info,
        get_git_branch_name,
        get_git_date,
        get_git_sha1,
    )
except ImportError:
    get_env_info = get_git_branch_name = get_git_date = get_git_sha1 = None

try:
    from .utils import (
        AttributeDict,
        MetricsTracker,
        add_eos,
        add_sos,
        concat,
        encode_supervisions,
        get_alignments,
        get_executor,
        get_texts,
        is_cjk,
        is_jit_tracing,
        is_module_available,
        l1_norm,
        l2_norm,
        linf_norm,
        load_alignments,
        make_pad_mask,
        measure_gradient_norms,
        measure_weight_norms,
        optim_step_and_measure_param_change,
        save_alignments,
        setup_logger,
        store_transcripts,
        str2bool,
        subsequent_chunk_mask,
        tokenize_by_CJK_char,
        tokenize_by_ja_char,
        write_error_stats,
    )
except ImportError:
    AttributeDict = MetricsTracker = add_eos = add_sos = concat = None
    encode_supervisions = get_alignments = get_executor = get_texts = is_cjk = None
    is_jit_tracing = is_module_available = l1_norm = l2_norm = linf_norm = None
    load_alignments = make_pad_mask = measure_gradient_norms = measure_weight_norms = None
    optim_step_and_measure_param_change = save_alignments = setup_logger = None
    store_transcripts = str2bool = subsequent_chunk_mask = tokenize_by_CJK_char = None
    tokenize_by_ja_char = write_error_stats = None

try:
    from .ngram_lm import NgramLm, NgramLmStateCost
except ImportError:
    NgramLm = NgramLmStateCost = None

try:
    from .lm_wrapper import LmScorer
except ImportError:
    LmScorer = None
