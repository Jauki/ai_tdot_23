import os
import shutil

def main():
    src = "D:\\newData3\\no_glasses"
    dest = "D:\\combined\\no_glasses"

    offset = len(os.listdir(dest))

    for file in os.listdir(src):
        shutil.copy(f'{src}\\{file}', f'{dest}\\{offset}.jpg')
        offset += 1

    pass

if __name__ == '__main__':
    main()