import pandas as pd,  matplotlib.pyplot as plt, numpy as np, re, itertools, os, sys
from cProfile import label
from matplotlib import patches
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from shapely.geometry.polygon import LinearRing, Point
from shapely import affinity
from textwrap import wrap
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import create_ellipse



def ellipse_polyline(ellipses, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result

def intersections(a, b):
    ea = LinearRing(a)
    eb = LinearRing(b)
    mp = ea.intersection(eb)
    try:
        x = [p.x for p in mp]
        y = [p.y for p in mp]
        return x, y, mp
    except:
        return [],[],[]

def plot_matrix(cm, labels):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
    Returns:
        summary: TensorFlow summary
    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(4, 4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    imcf = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=3,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    return fig

def outputCSV(dataInter,labels,pathFile):
    with open(pathFile,'w') as pf:        
        pf.write(','+','.join(labels) + '\n')
        for idx, d in enumerate(dataInter):
            pf.write(labels[idx]+',')
            pf.write(','.join(list(map(str,d)))+'\n')
    

def plotIntersections(mainElip,elips):
    fig,ax = plt.subplots()

    ##these next few lines are pretty important because
    ##otherwise your ellipses might only be displayed partly
    ##or may be distorted
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_aspect('equal')

    ##first ellipse in blue
    verts1 = np.array(mainElip.exterior.coords.xy)
    patch1 = patches.Polygon(verts1.T, color = 'blue', alpha = 0.5)
    ax.add_patch(patch1)

    for e in elips:
        if e is None:
            continue
        ##second ellipse in red    
        verts2 = np.array(e.exterior.coords.xy)
        patch2 = patches.Polygon(verts2.T,color = 'red', alpha = 0.5)
        ax.add_patch(patch2)

        ##the intersect will be outlined in black
        intersect = mainElip.intersection(e)
        verts3 = np.array(intersect.exterior.coords.xy)
        patch3 = patches.Polygon(verts3.T, facecolor = 'none', edgecolor = 'black')
        ax.add_patch(patch3)

    plt.show()

def calcIntersectionElipses(plotClasses,analiseItem=None):
    elipses = list(zip(plotClasses['valence mean'],plotClasses['arousal mean'],plotClasses['valence std'],plotClasses['arousal std'],[0] * len(plotClasses)))
    elipses = [create_ellipse(e[:2],e[2:-1]) for e in elipses]
    intersectionsMatrix = np.zeros( (len(elipses),len(elipses)) )
    for i in range(len(elipses)):
        for j in range(i+1,len(elipses)):
            if i == j:
                continue
            mp = elipses[i].intersection(elipses[j])
            intersectionsMatrix[i,j] = mp.area / elipses[i].area
            intersectionsMatrix[j,i] = mp.area / elipses[j].area
    

    #np.savetxt("test.csv",intersectionsMatrix,delimiter=',')
    sortedIntersections = (-intersectionsMatrix).argsort()
    classAndInters = {}
    for i in range(len(intersectionsMatrix)):
        classAndInters[plotClasses['class'][i]] = list(zip(intersectionsMatrix[i][sortedIntersections[i]],plotClasses['class'][sortedIntersections[i]]))

    if analiseItem is not None:
        plotIntersections(elipses[analiseItem],[elipses[i] if intersectionsMatrix[analiseItem][i] > 0 else None for i in (-intersectionsMatrix[analiseItem]).argsort()])

    #print('opa')

    '''
    with open('intersections.txt','w') as fileInt:
        for c in classAndInters:
            fileInt.write('%s -> ' % (c))
            for inter in classAndInters[c]:
                if inter[0] <=0 :
                    break
                fileInt.write("Class: %s intersection rating: %f | " % (inter[1],inter[0]))

            fileInt.write('\n')
    print('oi')
    '''
    '''
    elipses = ellipse_polyline(elipses)
    for i in range(len(elipses)):
        for j in range(len(elipses)):
            if i == j:
                continue
            x, y, mp = intersections(elipses[i],elipses[j])
            if len(x) > 2:
                fig,ax = plt.subplots()
                ax.set_xlim([-1,1])
                ax.set_ylim([-1,1])
                ax.set_aspect('equal')
                ax.plot(x, y, "o")
                ax.plot(elipses[i][:,0], elipses[i][:,1])
                ax.plot(elipses[j][:,0], elipses[j][:,1])
                plt.show()
                print('opa')
    '''
def plotClassesCircle(plotClasses):
    circles = list(zip(plotClasses['valence mean'],plotClasses['arousal mean']))
    distances = (plotClasses['valence std'] + plotClasses['arousal std']) / 2
    print(plotClasses)
    fig, ax = plt.subplots()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    for idx, c in enumerate(circles):
        ax.add_patch(plt.Circle(c,0.01, color='r',fill=False))
        ax.text(c[0],c[1],plotClasses['class'][idx])

    plt.xlim(-1, 1)
    plt.ylim(-1,1)
    plt.show()

def plotClassesEllipse(plotClasses):
    circles = list(zip(plotClasses['valence mean'],plotClasses['arousal mean']))
    print(plotClasses)
    fig, ax = plt.subplots()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    for idx, c in enumerate(circles):
        ax.add_patch(patches.Ellipse(c,plotClasses['valence std'][idx],plotClasses['arousal std'][idx],fill=False))
        ax.text(c[0],c[1],plotClasses['class'][idx])

    plt.xlim(-1, 1)
    plt.ylim(-1,1)
    plt.show()

def plotElbow(plotClasses):
    circles = np.array(list(zip(plotClasses['valence mean'],plotClasses['arousal mean'])))
    model = KMeans()
    visu = KElbowVisualizer(model,k=(2,12))
    visu.fit(circles)
    visu.show()

def groupEmotions(plotClasses):
    circles = np.array(list(zip(plotClasses['valence mean'],plotClasses['arousal mean'])))
    colors = ['red','lime','aqua','magenta']
    fig, ax = plt.subplots()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')

    kms = KMeans(n_clusters=4).fit(circles)
    labels = kms.labels_
    output = [[],[],[],[]]
    for idx, c in enumerate(plotClasses['class']):
        output[labels[idx]].append(c)
        ax.add_patch(plt.Circle(circles[idx],0.01, color=colors[labels[idx]],fill=False))
        ax.text(circles[idx][0],circles[idx][1],c)


    print(output)
    plt.xlim(-1, 1)
    plt.ylim(-1,1)
    plt.show()


if __name__ == "__main__":
    pc = pd.read_csv("hajer_categ.CSV")
    response = 1
    while response > 0:
        response = int(input("Type a number between 1 and 151 to show the graph of intersections: "))
        calcIntersectionElipses(pc,response - 1)