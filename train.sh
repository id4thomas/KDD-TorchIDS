#AE
# python main.py --batch_size 512 --epoch 10 --model ae


#VAE
# python main.py --batch_size 512 --epoch 10 --model vae --lr 1e-3

#MEMAE
# python main.py --batch_size 512 --epoch 10 --model memae

#MEMAE
# python main.py --batch_size 512 --epoch 10 --model memvae

#DAGMM
python main_dagmm.py --batch_size 1024 --epoch 20 --model dagmm --lr 1e-3
