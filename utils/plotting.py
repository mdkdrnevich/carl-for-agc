import uproot
import uproot.exceptions as exceptions
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def plot_distributions(nominal_data, alternate_data,
                       nominal_weights, carl_weights, alternate_weights,
                       nominal_mask=None, alternate_mask=None, carl_mask=None, alternate_name="", feature_name="", logscale=True, saveAs=None):
    font = font_manager.FontProperties(family='Symbol',
                                       style='normal', size=16)
    plt.rcParams['legend.title_fontsize'] = 18
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['figure.titlesize'] = 20

    hist_settings_nom = {'alpha': 0.25, 'color':'blue'}
    hist_settings_alt = {'alpha': 0.25, 'color':'orange'}
    hist_settings_CARL = {'histtype':'step', 'color':'black', 'linewidth':1, 'linestyle':'--'}
    hist_settings_CARL_ratio = {'color':'black', 'linewidth':1, 'linestyle':'--'}

    label = alternate_name
    legend = label
    legend_title = "Athena CARL"
    column = feature_name
    x_scaler = 1

    x0 = nominal_data
    w0 = nominal_weights
    w_carl = w0*carl_weights
    x1 = alternate_data
    w1 = alternate_weights
    
    if nominal_mask is not None or carl_mask is not None:
        athena_mask = np.zeros(w0.shape) == 0
        if nominal_mask is not None:
            athena_mask = athena_mask & nominal_mask(nominal_data)
        if carl_mask is not None:
            athena_mask = athena_mask & carl_mask(carl_weights)        
        print(athena_mask.sum())
        x0 = x0[athena_mask]
        w0 = w0[athena_mask]
        w_carl = w_carl[athena_mask]
    if alternate_mask is not None:
        athena_mask = alternate_mask(alternate_data)
        print(athena_mask.sum())
        x1 = x1[athena_mask]
        w1 = w1[athena_mask]

    # Normalize
    w0 /= w0.sum()
    w_carl /= w_carl.sum()
    w1 /= w1.sum()

    ### Start plotting

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, hspace=0, height_ratios=[5,2,2])
    axes = gs.subplots(sharex=True)

    #fig.suptitle("Differential Cross-section & Mapping Performance")

    binning = np.linspace(0, #min([np.percentile(x0, 0), np.percentile(x1, 0)]),
                          np.percentile(x0, 98),#min([np.percentile(x0, 98), np.percentile(x1, 98)]),
                          100)

    axes[0].hist(x0, bins=binning, weights=w0, label="Nominal", **hist_settings_nom, density=True)
    axes[0].hist(x0, bins=binning, weights=w_carl, label="Nominal*CARL", **hist_settings_CARL, density=True)
    axes[0].hist(x1, bins=binning, weights=w1, label=label, **hist_settings_alt, density=True);

    axes[0].grid(axis='x', color='silver')
    axes[0].set_title("Differential Cross-section & Mapping Performance")
    axes[0].set_xlabel('%s'%(column), horizontalalignment='right',x=1)
    axes[0].set_ylabel(r"$\frac{1}{N} \cdot \frac{d \sigma}{dx}$", horizontalalignment='center',x=1, fontsize=20)
    if logscale is True:
        axes[0].set_yscale("log")
    axes[0].legend(frameon=False,title = '%s sample'%(legend_title), prop=font )

    y_min, y_max = axes[0].get_ylim()
    axes[0].set_ylim([y_min*0.9, y_max*1.5])


    # ratio plot
    x0_hist, edge0 = np.histogram(x0, bins = binning, weights = w0, density=True)
    x1_hist, edge1 = np.histogram(x1, bins = binning, weights = w1, density=True)
    carl_hist, edgecarl = np.histogram(x0, bins = binning, weights = w_carl, density=True)
    #x1_ratio = x1_hist/x0_hist
    try:
        x1_ratio = x0_hist/x1_hist
    except ZeroDivisionError:
        x1_hist[x1_hist == 0] = np.nan
        x1_ratio = x0_hist/x1_hist
    #carl_ratio = carl_hist/x0_hist
    carl_ratio = carl_hist/x1_hist
    # Generate reference line
    #   -> Extract the lowest and highest bin edge
    xref= [binning.min(), binning.max()]
    #   -> Now generate the x and y points of the reference line
    yref = [1.0,1.0]

    ## Generate error bands and residue for the reference histogram
    x0_error = []
    x1_error = []
    residue = []
    residue_carl = []
    # Normalise weights to unity
    w0 = w0*(1.0/np.sum(w0))
    w1 = w1*(1.0/np.sum(w1))
    w_carl = w_carl*(1.0/np.sum(w_carl))
    if len(binning) > 1:
        width = abs(binning[1] - binning[0] )
        for xbin in binning:
            # Form masks for all event that match condition
            mask0 = (x0 < (xbin + width)) & (x0 > (xbin - width))
            mask1 = (x1 < (xbin + width)) & (x1 > (xbin - width))
            # Form bin error
            binsqrsum_x0 = np.sum(w0[mask0]**2)
            binsqrsum_x1 = np.sum(w1[mask1]**2)
            binsqrsum_x0_carl = np.sum(w_carl[mask0]**2)
            binsqrsum_x0 = math.sqrt(binsqrsum_x0)
            binsqrsum_x1 = math.sqrt(binsqrsum_x1)
            binsqrsum_x0_carl = math.sqrt(binsqrsum_x0_carl)
            # Form residue
            res_num = np.sum(w1[mask1]) - np.sum(w0[mask0])
            res_denom = math.sqrt(binsqrsum_x0**2 + binsqrsum_x1**2)
            # Form residue (CARL)
            res_num_carl = np.sum(w1[mask1]) - np.sum(w_carl[mask0])
            res_denom_carl = math.sqrt(binsqrsum_x0_carl**2 + binsqrsum_x1**2)
            # Form relative error
            try:
                binsqrsum_x0 = binsqrsum_x0/w0[mask0].sum()
            except ZeroDivisionError:
                pass
            try:
                binsqrsum_x1 = binsqrsum_x1/w1[mask1].sum()
            except ZeroDivisionError:
                pass

            # Save residual
            x0_error.append(binsqrsum_x0 if binsqrsum_x0 > 0 else 0.0)
            x1_error.append(binsqrsum_x1 if binsqrsum_x1 > 0 else 0.0)
            residue.append(res_num/res_denom if binsqrsum_x0+binsqrsum_x1 > 0 else 0.0)
            residue_carl.append(res_num_carl/res_denom_carl if binsqrsum_x0_carl+binsqrsum_x1 > 0 else 0.0)


    # Convert error lists to numpy arrays
    x0_error = np.array(x0_error)
    x1_error = np.array(x1_error)
    residue  = np.array(residue)
    residue_carl  = np.array(residue_carl)

    ## Ratio error
    axes[1].step( xref, yref, where="post", **hist_settings_alt )
    axes[1].step( edge1[:-1], x1_ratio, where="post", label="nom / "+legend, **hist_settings_nom)
    axes[1].step( edgecarl[:-1], carl_ratio, where="post", label = '(nominal*CARL) / '+legend, **hist_settings_CARL_ratio)
    axes[1].grid(axis='x', color='silver')
    yref_error = np.ones(len(edge1))
    yref_error_up = 2* np.sqrt( np.power(x1_error,2) + np.power(x0_error, 2)) # height from bottom
    yref_error_down = yref_error - np.sqrt(np.power(x1_error, 2) + np.power(x0_error,2))

    axes[1].bar( x=edge1[:-1],
                     height=yref_error_up[:-1], bottom = yref_error_down[:-1],
                     color='red',
                     width=np.diff(edge1),
                     align='edge',
                     alpha=0.25,
                     label='uncertainty band')


    axes[1].set_ylabel("Ratio", horizontalalignment='center',x=1)
    axes[1].set_xlabel('%s'%(column), horizontalalignment='right',x=1)
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_ylim([0.5, 1.6])
    axes[1].set_yticks(np.arange(0.5,1.6,0.1))


    ## Residual
    yref = [0.0,0.0]
    ref = axes[2].step( xref, yref, where="post", label=legend+" / "+legend, **hist_settings_alt )
    nom = axes[2].step( edge1, residue, where="post", label="nom / "+legend, **hist_settings_nom)
    carl = axes[2].step( edgecarl, residue_carl, where="post", label = '(nominal*CARL) / '+legend, **hist_settings_CARL_ratio)
    axes[2].grid(axis='x', color='silver')

    yref_error = np.zeros(len(edge1))
    yref_error_up = np.full(len(edge1), 1)
    yref_error_down = np.full(len(edge1), -1)
    yref_3error_up = np.full(len(edge1), 3)
    yref_3error_down = np.full(len(edge1), -3)
    yref_5error_up = np.full(len(edge1), 5)
    yref_5error_down = np.full(len(edge1), -5)

    FiveSigma = axes[2].fill_between(edge1, yref_5error_down, yref_5error_up, color='lightcoral', alpha=0.5, label = "5$\sigma$")
    ThreeSigma = axes[2].fill_between(edge1, yref_3error_down, yref_3error_up, color='bisque', alpha=0.75, label = "3$\sigma$")
    OneSigma = axes[2].fill_between(edge1, yref_error_down, yref_error_up, color='olivedrab', alpha=0.5, label = "1$\sigma$")

    axes[2].set_ylabel("Residual", horizontalalignment='center',x=1)
    axes[2].set_xlabel('%s'%(column), horizontalalignment='right',x=1)
    axes[2].legend(frameon=False,
                   ncol=3,
                   #title = '%s sample'%(label), 
                   handles=[OneSigma,ThreeSigma,FiveSigma],#,ref,nom,carl], 
                   labels = ["1$\sigma$", "3$\sigma$", "5$\sigma$"])#,("{} / {}").format(legend,legend), ("nom / {}").format(legend),("(nominal*CARL) / {}").format(legend)] )
    axes[2].set_ylim([-8, 8])
    axes[2].set_yticks(np.arange(-8,8,1.0));
    if saveAs is not None:
        fig.savefig(saveAs)