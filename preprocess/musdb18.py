import argparse













if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the spectrogram representation for the MUSDB18 dataset "
                                                 "Specifically, the dataset must be in .wav format")
    parser.add_argument("input_path", help='The folder path for the MUSDB18 dataset')
    parser.add_argument("output_path", help='Specify the folder where you want to store the created spectrograms.')
    args = parser.parse_args()
