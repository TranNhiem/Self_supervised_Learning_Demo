import os, shutil

train_path = '/img_data/train'
one_per_path = '/img_data/one_per/dataset/train'
ten_per_path = '/img_data/ten_per/dataset/train'

class_names = [x for x in os.listdir(train_path) if '.tar' not in x]

for class_name in class_names:
    if not os.path.exists(os.path.join(one_per_path, class_name)):
        os.mkdir(os.path.join(one_per_path, class_name))
    if not os.path.exists(os.path.join(ten_per_path, class_name)):
        os.mkdir(os.path.join(ten_per_path, class_name))

with open('/code_spec/downstream_tasks/one_per.txt', 'r') as f:
    one_per_images = f.readlines()
    for image in one_per_images:
        image = image[:-1]
        label, _ = image.split('_')
        src_path = os.path.join(train_path, label, image)
        dest_path = os.path.join(one_per_path, label, image)
        if not os.path.exists(dest_path) and os.path.exists(src_path):
            shutil.copyfile(src_path, dest_path)

with open('/code_spec/downstream_tasks/ten_per.txt', 'r') as f:
    ten_per_images = f.readlines()
    for image in ten_per_images:
        image = image[:-1]
        label, _ = image.split('_')
        src_path = os.path.join(train_path, label, image)
        dest_path = os.path.join(ten_per_path, label, image)
        if not os.path.exists(dest_path) and os.path.exists(src_path):
            shutil.copyfile(src_path, dest_path)
