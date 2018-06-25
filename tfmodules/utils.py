import sys


def progress_bar(total, progress, state_msg):
    """
    Displays or updates a console progress bar.
    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"

    block = int(round(barLength * progress))
    progress_bar = "\r [{}] {:.0f}% -> {}{}".format("#" * block + "-" * (barLength - block),
                                                    round(progress * 100, 0),
                                                    state_msg,
                                                    status)
    sys.stdout.write(progress_bar)
    sys.stdout.flush()
