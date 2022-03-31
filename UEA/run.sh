#!bin/bash

## UEA
if [ "$#" -lt 2 ]; then
  echo "Please select datasets"
  exit 1
fi

args=("$@")

echo "GPU: ${1}"

for (( c=1; c<$#; c++ ))
do
  if [ "${args[$c]}" -eq 1 ]; then
    python main.py -g=${1} --dataset=ArticularyWordRecognition --time=2 --kernels=10,10,10
  fi
  if [ "${args[$c]}" -eq 2 ]; then
    python main.py -g=${1} --dataset=AtrialFibrillation --time=5 --kernels=64,1,1 --filters=64,64,64 --matrix_dim=8 --rnn_dim=16
  fi
  if [ "${args[$c]}" -eq 3 ]; then
    python main.py -g=${1} --dataset=BasicMotions
  fi
  if [ "${args[$c]}" -eq 4 ]; then
    python main.py -g=${1} --dataset=CharacterTrajectories --time=5 --kernels=2,2,2
  fi
  if [ "${args[$c]}" -eq 5 ]; then
    python main.py -g=${1} --dataset=Cricket --time=7 --kernels=50,1,1
  fi
  if [ "${args[$c]}" -eq 6 ]; then
    python main.py -g=${1} --dataset=DuckDuckGeese --time=2 --kernels=15,15,15
  fi
  if [ "${args[$c]}" -eq 7 ]; then
    python main.py -g=${1} --dataset=EigenWorms --time=30 --kernels=150,10,10 # 3
  fi
  if [ "${args[$c]}" -eq 8 ]; then
    python main.py -g=${1} --dataset=Epilepsy --time=5 --kernels=2,2,2 # 4
  fi
  if [ "${args[$c]}" -eq 9 ]; then
    python main.py -g=${1} --dataset=EthanolConcentration --time=20 --kernels=50,2,2 --filters=16,16,16 # 5
  fi
  if [ "${args[$c]}" -eq 10 ]; then
    python main.py -g=${1} --dataset=ERing --kernels=1,1,1
  fi
  if [ "${args[$c]}" -eq 11 ]; then
    python main.py -g=${1} --dataset=FaceDetection --time=2 --kernels=20,1,1
  fi
  if [ "${args[$c]}" -eq 12 ]; then
    python main.py -g=${1} --dataset=FingerMovements --time=5 --kernels=2,2,2 --filters=8,8,8 --matrix_dim=16
  fi
  if [ "${args[$c]}" -eq 13 ]; then
    python main.py -g=${1} --dataset=HandMovementDirection --time=10 --kernels=2,2,2 --matrix_dim=16 --rnn_dim=16 --filters=128,128,128
  fi
  if [ "${args[$c]}" -eq 14 ]; then
    python main.py -g=${1} --dataset=Handwriting --time=2 --kernels=50,1,1
  fi
  if [ "${args[$c]}" -eq 15 ]; then
    python main.py -g=${1} --dataset=Heartbeat --time=10 --kernels=2,2,2 --lr=5e-4 --filters=32,32,32 --matrix_dim=16
  fi
  if [ "${args[$c]}" -eq 16 ]; then
    python main.py -g=${1} --dataset=InsectWingbeat --time=5 --kernels=2,2,2
  fi
  if [ "${args[$c]}" -eq 17 ]; then
    python main.py -g=${1} --dataset=JapaneseVowels --time=2 --kernels=1,1,1
  fi
  if [ "${args[$c]}" -eq 18 ]; then
    python main.py -g=${1} --dataset=Libras --time=3 --kernels=2,2,2 --matrix_dim=8 --rnn_dim=8 --filters=32,32,32
  fi
  if [ "${args[$c]}" -eq 19 ]; then
    python main.py -g=${1} --dataset=LSST --time=2 --kernels=5,1,1
  fi
  if [ "${args[$c]}" -eq 20 ]; then
    python main.py -g=${1} --dataset=MotorImagery --time=20 --kernels=10,10,10
  fi
  if [ "${args[$c]}" -eq 21 ]; then
    python main.py -g=${1} --dataset=NATOPS --time=5 --kernels=2,2,2
  fi
  if [ "${args[$c]}" -eq 22 ]; then
    python main.py -g=${1} --dataset=PenDigits --time=2 --kernels=2,2,2 --matrix_dim=64 --rnn_dim=64
  fi
  if [ "${args[$c]}" -eq 23 ]; then
    python main.py -g=${1} --dataset=PEMS-SF --time=5 --kernels=2,2,2 --matrix_dim=16
  fi
  if [ "${args[$c]}" -eq 24 ]; then
    python main.py -g=${1} --dataset=PhonemeSpectra --time=5 --kernels=10,2,2
  fi
  if [ "${args[$c]}" -eq 25 ]; then
    python main.py -g=${1} --dataset=RacketSports --time=2 --kernels=5,5,5 # --time=10 --kernels=1,1,1
  fi
  if [ "${args[$c]}" -eq 26 ]; then
    python main.py -g=${1} --dataset=SelfRegulationSCP1 --time=10 --kernels=16,16,16
  fi
  if [ "${args[$c]}" -eq 27 ]; then
    python main.py -g=${1} --dataset=SelfRegulationSCP2 --time=5 --kernels=64,2,2 --dilation=2 --lr=5e-4
  fi
  if [ "${args[$c]}" -eq 28 ]; then
    python main.py -g=${1} --dataset=SpokenArabicDigits --time=5 --kernels=2,2,2 --matrix_dim=16 --rnn_dim=16
  fi
  if [ "${args[$c]}" -eq 29 ]; then
    python main.py -g=${1} --dataset=StandWalkJump --time=10 --kernels=250,1,1 --filters=16,16,16 --matrix_dim=16 --rnn_dim=8 --lr=1e-4 --weight_decay=5e-5
  fi
  if [ "${args[$c]}" -eq 30 ]; then
    python main.py -g=${1} --dataset=UWaveGestureLibrary --time=5 --kernels=10,10,10 --lr=1e-3
  fi
done
