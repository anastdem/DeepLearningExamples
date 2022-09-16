!/usr/bin/env bash

set -e

DATADIR="data/RUSLAN22"
FILELISTSDIR="filelists"

TRAINLIST="$FILELISTSDIR/ruslan_audio_text_train_emp.txt"
VALLIST="$FILELISTSDIR/ruslan_audio_text_val_emp.txt"

TRAINLIST_MEL="$FILELISTSDIR/ruslan_mel_text_train_emp.txt"
VALLIST_MEL="$FILELISTSDIR/ruslan_mel_text_val_emp.txt"


if [ $(ls $DATADIR/mels | wc -l) -ne 13100 ]; then
    python preprocess_audio2mel.py --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL"
    python preprocess_audio2mel.py --wav-files "$VALLIST" --mel-files "$VALLIST_MEL"	
fi	
