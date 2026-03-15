# Copyright (c) ModelScope Contributors. All rights reserved.

if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    from swift.cli.main import prepare_config_args
    prepare_config_args(argv)
    sys.argv = [sys.argv[0]] + argv
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.pipelines import rlhf_main
    rlhf_main()
