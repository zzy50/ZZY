
cd coco

# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

# 이상하게 sh로 실행하면 안되고 명령어 그대로 복붙하면 됨
# 그냥 복붙해서 쓰자