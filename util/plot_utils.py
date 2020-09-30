import matplotlib.pyplot as plt

def plot_losses(train_l,val_l,title="Plot"):
    fig=plt.figure()
    l_plot=fig.add_subplot(1,1,1)
    l_plot.plot(range(len(train_l)),train_l,c='r',label='train')
    l_plot.plot(range(len(val_l)),val_l,c='b',label='val')
    l_plot.set_title(title)
    return fig
