# StereoMatching
Python program for a stereo analysis system involving feature-based, region-based and multi-resolution matching.

Stereo Matching is done using two methods. Namely, Region based, and feature based. The matching scores available are SSD, SAD and NCC. Either can be selected in the terminal.
Steps:
• At a given level, stereo matching is done, and disparity calculated.
• Validity check is then performed, wherein if the left-to-right match does not correspond
to right-to-left match, a zero is placed at that location in the disparity.
• Averaging is performed in the neighborhood to fill these zeroes.
• Disparity is propagated to the next lower (finer) level. This is done by duplicating
disparity from 1-pixel to the corresponding 4-pixels in the lower level.
• Using this disparity as starting point of the search, stereo matching is performed, and
disparity is updated.
• Harris corners detection is used for feature detection and Harris corner response
measure is used as the descriptor value for matching.
