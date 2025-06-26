class GTA5(Dataset):
    def __init__(self, rootdir, file_names, imgdir="images", targetdir="labels", augment=False, transform=None, target_transform=None, rcs=True, rcs_temp=0.018):
        super(GTA5, self).__init__()

        self.rcs = rcs

        if self.rcs:
          self.rcs_temp = rcs_temp

          self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(self.rcs_temp)
          self.samples_with_class = samples_with_class()

        self.to_tensor = v2.ToTensor()
        self.to_pil = v2.ToPILImage()

        self.gaussian_blur = v2.GaussianBlur(7)

        s = 0.6
        self.color_jitter = v2.ColorJitter(brightness=s, contrast=s, saturation=s, hue=0.1)
        self.gaussian_noise = v2.GaussianNoise(mean=0.0, sigma=0.4)

        self.augmentations = {
            self.gaussian_blur: 0.5,
            self.color_jitter: 0.5,
            self.gaussian_noise: 0.5
        }

        self.rootdir = rootdir

        self.targetdir = os.path.join(self.rootdir, targetdir) # ./labels
        self.imgdir = os.path.join(self.rootdir, imgdir) # ./images

        self.augment = augment

        self.transform = transform
        self.target_transform = target_transform

        self.imgs_path = []
        self.targets_color_path = []
        self.targets_labelIds_path = []

        for image_file in file_names: # 00001.png
            self.imgs_path.append(os.path.join(self.imgdir, image_file)) #./images/00001.png

            target_color_path = image_file # 00001.png
            target_labelsId_path = image_file.split(".")[0]+"_labelIds.png" # 00001_labelIds.png

            self.targets_color_path.append(os.path.join(self.targetdir, target_color_path)) #./labels/00001.png
            self.targets_labelIds_path.append(os.path.join(self.targetdir, target_labelsId_path)) #./labels/00001_labelIDs.png

    def create_target_img(self):
        list_ = GTA5Labels_TaskCV2017().list_

        for i, img_path in tqdm(enumerate(self.targets_color_path)):
            image_numpy = np.asarray(Image.open(img_path).convert('RGB'))

            H, W, _ = image_numpy.shape
            label_image = 255*np.ones((H, W), dtype=np.uint8)

            for label in list_:
                label_image[(image_numpy == label.color).all(axis=-1)] = label.ID

            new_img = Image.fromarray(label_image)
            new_img.save(self.targets_labelIds_path[i])

    def __getitem__(self, idx):
        if self.rcs == True:
          image_file = self._pick_rare_class_sample() # e.g. 00001.png
          image = Image.open("Gta5_extended/images/"+image_file).convert('RGB')
          target_color = Image.open("Gta5_extended/labels/"+image_file).convert('RGB')
          target_labelIds = cv2.imread("Gta5_extended/labels/"+image_file.split('.')[0]+"_labelIds.png", cv2.IMREAD_UNCHANGED).astype(np.int64)
        else:
          image = Image.open(self.imgs_path[idx]).convert('RGB')
          target_color = Image.open(self.targets_color_path[idx]).convert('RGB')
          target_labelIds = cv2.imread(self.targets_labelIds_path[idx], cv2.IMREAD_UNCHANGED).astype(np.int64)

        if self.augment:
            image, target_color, target_labelIds = self.augment_data(image, target_color, target_labelIds)

        if self.transform is not None:
            image = self.transform(image)
            target_color = self.transform(target_color)

        if self.target_transform is not None:
            target_labelIds = self.target_transform(target_labelIds)

        return image, target_color, target_labelIds

    def __len__(self):
        return len(self.imgs_path)

    def augment_data(self, image, target_color, target_labelIds):
        image = self.to_tensor(image)

    # Geometric Transformations
        # Horizontal Flip
        # img_tensor = v2.functional.hflip(img_tensor)
        # target_color = np.fliplr(target_color).copy()
        # target_labelIds = np.fliplr(target_labelIds).copy()

        for augmentation, p in self.augmentations.items():
            if random.random() < p:
                image = augmentation(image)

        image = torch.clamp(image, 0.0, 1.0)
        image = self.to_pil(image)

        return image, target_color, target_labelIds

    def _pick_rare_class_sample(self):
      cls = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
      return np.random.choice(self.samples_with_class[cls])


def GTA5_dataset_splitter(rootdir, train_split_percent, split_seed = None, imgdir="images", targetdir="labels", augment=False, rcs=False, rcs_temp=0.018, transform=None, target_transform=None):
    assert 0.0 <= train_split_percent <= 1.0, "train_split_percent should be a float between 0 and 1"

    target_path = os.path.join(rootdir, targetdir) # ./labels
    img_path = os.path.join(rootdir, imgdir) # ./images

    file_names = [
        f for f in os.listdir(img_path)
        if f.endswith(".png") and os.path.exists(os.path.join(target_path, f.split(".")[0]+"_labelIds.png"))
    ]

    if split_seed is not None:
        random.seed(split_seed)
    random.shuffle(file_names)
    random.seed()

    split_idx = int(len(file_names) * train_split_percent)

    train_files = file_names[:split_idx]
    val_files = file_names[split_idx:]

    return GTA5(rootdir, train_files, imgdir, targetdir, augment, transform, target_transform, rcs=rcs, rcs_temp=rcs_temp), \
           GTA5(rootdir, val_files, imgdir, targetdir, False,  transform, target_transform, rcs=False)