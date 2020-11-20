import argparse
import torch
import torch.nn.functional as tnf
from moviepy.editor import VideoFileClip
from torchvision import transforms


class MakeMask:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        self.people_class = 15

        self.model.eval()
        print("Model Loaded")

        self.blur = torch.FloatTensor([[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]) / 16.0

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.model.to('cuda')
            self.blur = self.blur.to('cuda')

        self.preprocess = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    def __call__(self, img):
        frame_data = torch.FloatTensor(img) / 255.0

        input_tensor = self.preprocess(frame_data.permute(2, 0, 1))
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        segmentation = output.argmax(0)

        bgout = output[0:1][:][:]
        a = (1.0 - tnf.relu(torch.tanh(bgout * 0.30 - 1.0))).pow(0.5) * 2.0

        people = segmentation.eq(torch.ones_like(segmentation).long().fill_(self.people_class)).float()

        people.unsqueeze_(0).unsqueeze_(0)

        for i in range(3):
            people = tnf.conv2d(people, self.blur, stride=1, padding=1)

        # combined_mask = tnf.hardtanh(a * b)
        combined_mask = tnf.relu(tnf.hardtanh(a * (people.squeeze().pow(1.5))))
        combined_mask = combined_mask.expand(1, 3, -1, -1)

        res = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

        return res


def processmovie(args):
    if args.width != 0:
        target = [args.width, None]
    else:
        target = None

    realityclip = VideoFileClip(args.input, target_resolution=target)

    masked = realityclip.fl_image(MakeMask())
    masked.write_videofile(args.output)


def cli():
    parser = argparse.ArgumentParser(description='BGRemove')
    parser.add_argument('--input', metavar='N', required=True, help='input movie path')
    parser.add_argument('--output', metavar='N', required=True, help='output movie path')
    parser.add_argument('--width', metavar='N', type=int, default=0,
                        help='target width (optional, omit for full width)')

    args = parser.parse_args()

    processmovie(args)


if __name__ == '__main__':
    cli()
