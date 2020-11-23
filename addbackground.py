from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
import argparse

parser = argparse.ArgumentParser(description='GSE')
parser.add_argument('--input', metavar='N', required=True, help='input movie path')
parser.add_argument('--bg', metavar='N', required=True, help='new background path')
parser.add_argument('--mask', metavar='N', required=True, help='mask movie/image path')
parser.add_argument('--output', metavar='N', required=True, help='output movie path')

args = parser.parse_args()

real = VideoFileClip(args.input)

themask = VideoFileClip(args.mask).to_mask()

new = real.set_mask(themask)

background = ImageClip(args.bg, duration=new.duration)

final = CompositeVideoClip([background, new], size=new.size)

final.write_videofile(args.output, threads=4)