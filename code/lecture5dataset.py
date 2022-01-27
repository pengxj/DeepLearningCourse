import torch
import torch.utils.data as data
from PIL import Image
import os


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def video_loader(video_dir_path, frame_indices):
    video = []
    for i in frame_indices: #[0, 1, 2]:
        image_path = video_dir_path + f'_{i}.jpg'
        if os.path.exists(image_path):
            video.append(pil_loader(image_path))
        else:
            return []

    return video


def get_video_names_and_annotations(root_path):
    videos = []
    annotations = []
    pos_cnt = 0
    neg_cnt = 0
    org_folders = os.listdir(root_path)
    n_org_samples = len(org_folders)
    for i in range(n_org_samples):
        if os.path.exists(os.path.join(root_path, org_folders[i], 'pos')):
            v_path = os.path.join(root_path, org_folders[i], 'pos')
            segments = [name.split('_')[0] for name in os.listdir(v_path)]
            unique_segments = list(set(segments))
            for s in unique_segments:
                videos.append(os.path.join(v_path, s))
                annotations.append(1)
                pos_cnt += 1

        if os.path.exists(os.path.join(root_path, org_folders[i], 'neg')):
            v_path = os.path.join(root_path, org_folders[i], 'neg')
            segments = [name.split('_')[0] for name in os.listdir(v_path)]
            unique_segments = list(set(segments))
            for s in unique_segments:
                videos.append(os.path.join(v_path, s))
                annotations.append(0)
                neg_cnt += 1
    print(f'positive samples: {pos_cnt}, negative samples: {neg_cnt}')
    return videos, annotations


def make_dataset(root_path):
    video_paths, annotations = get_video_names_and_annotations(root_path)
    dataset = []
    for i in range(len(video_paths)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_paths)))
        video_path = video_paths[i]
        clip = video_loader(video_path, [0, 1, 2])
        if clip:
            sample = {'video': video_path}
            sample['label'] = annotations[i]
            dataset.append(sample)
        else:
            print(f'video: {video_path} can not be loaded.')

    return dataset


class VideoSmoke(data.Dataset):
    def __init__(self,
                 root_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None):
        self.data = make_dataset(root_path)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = video_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = [i for i in range(3)] #0,1,2
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.loader(path, frame_indices)

        if self.spatial_transform is not None:
            # self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0)#.permute(1, 0, 2, 3) # C T H W

        target = self.data[index]['label']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from torchvision import transforms

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(128),
            transforms.RandomCrop(112),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    vsmoke = VideoSmoke('data', spatial_transform=data_transforms['train'])
    loader = data.DataLoader(vsmoke, batch_size=1)
    iter_loader = iter(loader)
    print(next(iter_loader))