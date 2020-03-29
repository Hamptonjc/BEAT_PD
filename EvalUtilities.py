from sklearn.metrics import auc, roc_curve
from sklearn.utils import check_matplotlib_support

class RocCurveDisplay:
    """ROC Curve visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_roc_curve` to create a
    visualizer. All parameters are stored as attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    fpr : ndarray
        False positive rate.
    tpr : ndarray
        True positive rate.
    roc_auc : float
        Area under ROC curve.
    estimator_name : str
        Name of estimator.
    Attributes
    ----------
    line_ : matplotlib Artist
        ROC Curve.
    ax_ : matplotlib Axes
        Axes with ROC Curve.
    figure_ : matplotlib Figure
        Figure containing the curve.
    """

    def __init__(self, fpr, tpr, roc_auc, estimator_name):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.estimator_name = estimator_name

    def plot(self, ax=None, name=None, **kwargs):
        """Plot visualization
        Extra keyword arguments will be passed to matplotlib's ``plot``.
        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use the name of the
            estimator.
        Returns
        -------
        display : :class:`~sklearn.metrics.plot.RocCurveDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support('RocCurveDisplay.plot')
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        name = self.estimator_name if name is None else name

        line_kwargs = {
            'label': "{} (AUC = {:0.2f})".format(name, self.roc_auc)
        }
        line_kwargs.update(**kwargs)

        self.line_ = ax.plot(self.fpr, self.tpr, **line_kwargs)[0]
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right')

        self.ax_ = ax
        self.figure_ = ax.figure
        return self
    
import matplotlib.pyplot as plt 

def plot_ROC(fpr, tpr, roc_auc, title = 'Binary Classifier', 
             legend = "Conv Neural Net" ):
    '''plot results from sklearn roc_curve command
    
    Intended usage given yTrue, yPred as actual, predicted
    iterables with the same len
    
    -----------------------------------------
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(yTrue, yPred)
    roc_auc = auc(fpr,tpr)
    plot_ROC(fpr,tpr, roc_auc)
    
    option: 
    ==========
    
    title = Title of plot (default = 'Binary Classifier')
    legend = name of classifier (default = "Conv Neural Net")
    
    '''
    fig,ax = plt.subplots(figsize = (6,6))
    viz = RocCurveDisplay(fpr, tpr, roc_auc, legend)
    viz.plot(ax=ax)
    ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
    ax.set_xlim(-0.05,1)
    ax.set_ylim(0,1.05)
    ax.set_xlabel('False Positive Rate', fontsize = 14)
    ax.set_ylabel('True Positive Rate', fontsize = 14)
    ax.set_title( title )
    ax.legend(fontsize = 12);
    return viz   
    
## Scoring 

def BEATPDscoring( PredictionsDf ):
    '''Scores predictions saved in a data frame. Required format:
    
    PredictionsDf.columns = ['measurement_id', 'subject_id', 'actual','predicted', ...]
    
    NOTE: columns can be in any order. Ellipsis ... indicates other columns are ignored
    
    returns MSEdf.columns = ['$n_k$', '$MSE_k$', '$\frac{\sqrt{n_k} {MSE}_k}{\sum_{k=1}^N \sqrt{n_k} }$']
    
    FinalScore_sqrt = MSEdf.iloc[:,-1].sum() 
    
    '''
    sub_ids = PredictionsDf.subject_id.unique()
    cols = ['$n_k$','$MSE_k$']
    MSEdf = pd.DataFrame( np.zeros( (len(sub_ids),2) ), 
                          columns = cols, index = sub_ids)
    for sub_id in sub_ids:
        subDf = PredictionsDf.loc[ PredictionsDf.subject_id == sub_id ]
        MSEdf.loc[sub_id, '$MSE_k$'] = ((subDf['actual'] - subDf['predicted'])**2).mean()
        MSEdf.loc[sub_id, '$n_k$'] = len(subDf)
    sqrtnkTotal = np.sqrt(MSEdf['$n_k$']).sum()
    colName = r'$\frac{\sqrt{n_k} {MSE}_k}{\sum_{k=1}^N \sqrt{n_k} }$'
    MSEdf[colName] = np.sqrt(MSEdf['$n_k$']) * MSEdf['$MSE_k$']
    return MSEdf    
    
print("plot_ROC and BEATPDscoring commands now loaded.")  
    