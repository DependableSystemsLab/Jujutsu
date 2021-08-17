from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--nsampling', type=int, default=50, help='ramplimg # times for each images')
        parser.add_argument('--save_number', type=int, default=10, help='choice # reasonable results based on the discriminator score')
        parser.add_argument('--mask_pert', type=float, default=0.25, help='pert of pixel to mask')
        parser.add_argument('--is_square_patch', type=int, default=1, help='indicate a square region to mask')


        self.isTrain = False

        return parser

