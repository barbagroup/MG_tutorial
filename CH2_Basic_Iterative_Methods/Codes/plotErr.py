def plotError(err, title, lab, Ltype, log):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    plt.figure(title)
    if log == 0:
        plt.ylim((0, 1))
    plt.xlim((0, 100))
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Error (Max. norm)')

    if log == 0:
        ax = plt.gca()
        Xmajor = MultipleLocator(25)
        Ymajor = MultipleLocator(0.25)
        Xminor = MultipleLocator(12.5)
        Yminor = MultipleLocator(0.125)

        ax.xaxis.set_major_locator(Xmajor)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(Xminor)

        ax.yaxis.set_major_locator(Ymajor)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%4.2f'))
        ax.yaxis.set_minor_locator(Yminor)

    plt.grid(True, which='both')

    if len(lab) == 1:
        if log == 0:
            plt.plot(np.array(range(101)), err,
                     Ltype[0], lw=2, label=lab[0])
        else:
            plt.semilogy(np.array(range(101)), err,
                         Ltype[0], lw=2, label=lab[0],
                         basey=10)
    else:
        if log == 0:
            for i, text in enumerate(lab):
                plt.plot(np.array(range(101)), err[i, :],
                         Ltype[i], lw=2, label=text)
        else:
            for i, text in enumerate(lab):
                plt.semilogy(np.array(range(101)), err[i, :],
                             Ltype[i], lw=2, label=text,
                             basey=np.e)

    plt.legend(loc=0)
