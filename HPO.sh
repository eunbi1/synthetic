#rm HPO_log;
for score_type in "professor_suggestion"; #"professor_suggestion" "rectified_with_hyperparam" "real+linear" "rectified+linear";
do
  for alpha in 2.0 1.9 1.8 1.7 1.6 1.5 1.4 1.3 1.2 1.1;
  do
    python HPO.py --alpha ${alpha} --score_type ${score_type} > HPO_log_${alpha}_${score_type} 2>&1 &
  done;
done