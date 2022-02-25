DATA_DIR=/workspace/FasterTransformer/bert-quantization/bert-pyt-quantization/model/bert-base-cased-squad2

python3 inference.py --bert_model bert-base-uncased \
    --init_checkpoint $DATA_DIR/pytorch_model.bin \
    --vocab_file $DATA_DIR/vocab.txt \
    --config_file $DATA_DIR/config.json \
    --n_best_size 5 \
    --max_answer_length 50 \
    --question "What's the nickname of Babe Ruth Jr?" \
    --context "George Herman 'Babe' Ruth Jr was an American professional baseball player whose career in Major League Baseball (MLB) spanned 22 seasons, from 1914 through 1935. Nicknamed 'The Bambino' and 'The Sultan of Swat', he began his MLB career as a star left-handed pitcher for the Boston Red Sox, but achieved his greatest fame as a slugging outfielder for the New York Yankees. Ruth is regarded as one of the greatest sports heroes in American culture and is considered by many to be the greatest baseball player of all time. In 1936, Ruth was elected into the Baseball Hall of Fame as one of its 'first five' inaugural members." \