ffmpeg -framerate 10 -i ${2}/${1}-out/%05d.png -pix_fmt yuv420p ${2}/${1}-optim.mp4
