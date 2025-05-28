import mne
import matplotlib.pyplot as plt


def main():
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    fig = montage.plot(kind="3d", show=False)
    ax = fig.gca()
    ax.view_init(elev=15, azim=70)
    plt.show()


if __name__ == "__main__":
    main()
