import numpy as np
#model_a=[np.float64(0.000684978615026921), np.float64(0.0006489958032034338), np.float64(0.0007631533080711961), np.float64(0.0006735915085300803), np.float64(0.0006028122152201831), np.float64(0.0008132510120049119), np.float64(0.0005208284710533917), np.float64(0.0006062673055566847), np.float64(0.0006620751810260117), np.float64(0.0007557764765806496), np.float64(0.0007232788484543562), np.float64(0.0007777598220854998), np.float64(0.0006978213205002248), np.float64(0.0006652381853200495), np.float64(0.000545329530723393), np.float64(0.0005525872693397105), np.float64(0.0005627571954391897), np.float64(0.000528015021700412), np.float64(0.000590028241276741), np.float64(0.0007275560055859387), np.float64(0.0005793251912109554), np.float64(0.0006057932041585445), np.float64(0.0007291259826160967), np.float64(0.0006421868456527591), np.float64(0.0003837484691757709), np.float64(0.0005842732498422265), np.float64(0.0006652179290540516), np.float64(0.0005984315066598356), np.float64(0.0007186622242443264), np.float64(0.0006385635933838785), np.float64(0.0006773899658583105), np.float64(0.0006959204911254346), np.float64(0.0007190159521996975), np.float64(0.0006990838446654379), np.float64(0.0007005406077951193), np.float64(0.0007191619952209294), np.float64(0.0005697711021639407), np.float64(0.0006035229307599366), np.float64(0.0006652424344792962), np.float64(0.0008817475754767656), np.float64(0.0007832135306671262), np.float64(0.0006841491558589041), np.float64(0.000803012284450233), np.float64(0.0007963391835801303), np.float64(0.0006382956635206938), np.float64(0.0007447742391377687), np.float64(0.000733728229533881), np.float64(0.000737069349270314), np.float64(0.0005527837201952934), np.float64(0.0007425050134770572)]
#model_b= [np.float64(0.0002586366899777204), np.float64(0.0003076068533118814), np.float64(0.00041860633064061403), np.float64(0.0007428214885294437), np.float64(0.0003501062165014446), np.float64(0.0003605330130085349), np.float64(0.00032315001590177417), np.float64(0.0002797054185066372), np.float64(0.00036372977774590254), np.float64(0.0002965026651509106), np.float64(0.00039702720823697746), np.float64(0.00035725871566683054), np.float64(0.0003464265028014779), np.float64(0.000402188888983801), np.float64(0.0004190526087768376), np.float64(0.0002959785342682153), np.float64(0.00035673510865308344), np.float64(0.0004090467991773039), np.float64(0.0003310787142254412), np.float64(0.0003081663162447512), np.float64(0.0003224829852115363), np.float64(0.00027495689573697746), np.float64(0.0003355587541591376), np.float64(0.00038941114326007664), np.float64(0.0003026232006959617), np.float64(0.000347134773619473), np.float64(0.0003049213264603168), np.float64(0.0003383138100616634), np.float64(0.0003243648970965296), np.float64(0.00034960173070430756), np.float64(0.00027713042800314724), np.float64(0.0002878782688640058), np.float64(0.00035151815973222256), np.float64(0.00034349309862591326), np.float64(0.00041142874397337437), np.float64(0.00037434836849570274), np.float64(0.00033462134888395667), np.float64(0.0004234712978359312), np.float64(0.0003096860891673714), np.float64(0.00036697715404443443), np.float64(0.00028042669873684645), np.float64(0.0003592275024857372), np.float64(0.00036231050034984946), np.float64(0.0003234267351217568), np.float64(0.000309688359266147), np.float64(0.00035863014636561275), np.float64(0.0005491954507306218), np.float64(0.00031977385515347123), np.float64(0.0004019199695903808), np.float64(0.00032750010723248124)]
from scipy import stats
model_a=  [np.float64(35.4988), np.float64(26.374), np.float64(37.345), np.float64(35.3587), np.float64(39.007), np.float64(37.8521), np.float64(23.5146), np.float64(28.1543), np.float64(35.7928), np.float64(24.0939), np.float64(38.3885), np.float64(17.8158), np.float64(27.2669), np.float64(40.1694), np.float64(34.9977), np.float64(23.9235), np.float64(37.8167), np.float64(39.2788), np.float64(37.47), np.float64(26.726), np.float64(37.1755), np.float64(21.7201), np.float64(25.7272), np.float64(20.1667), np.float64(27.4812), np.float64(35.8572), np.float64(27.7525), np.float64(38.7338), np.float64(27.5563), np.float64(32.2279), np.float64(35.7856), np.float64(19.8539), np.float64(29.2698), np.float64(34.9061), np.float64(18.1643), np.float64(39.2166), np.float64(35.5464), np.float64(29.9752), np.float64(34.1942), np.float64(38.9363), np.float64(31.2071), np.float64(37.2047), np.float64(29.5562), np.float64(31.8566), np.float64(36.5502), np.float64(32.8627), np.float64(28.3282), np.float64(19.7482), np.float64(34.6914), np.float64(40.7872)]
model_b = [np.float64(22.0321), np.float64(21.4323), np.float64(15.8672), np.float64(19.9514), np.float64(27.3396), np.float64(19.7577), np.float64(24.6294), np.float64(23.1095), np.float64(24.662), np.float64(38.8472), np.float64(17.077), np.float64(19.5714), np.float64(18.1697), np.float64(18.2768), np.float64(20.8474), np.float64(48.7284), np.float64(17.9758), np.float64(15.8408), np.float64(52.9763), np.float64(19.6627), np.float64(19.667), np.float64(19.9707), np.float64(17.3898), np.float64(21.9421), np.float64(19.0571), np.float64(20.7315), np.float64(23.6143), np.float64(31.5565), np.float64(23.5871), np.float64(18.8005), np.float64(48.12), np.float64(23.1788), np.float64(19.3778), np.float64(19.0922), np.float64(16.7637), np.float64(17.5198), np.float64(24.2654), np.float64(24.1306), np.float64(18.1478), np.float64(15.4895), np.float64(16.8799), np.float64(18.078), np.float64(19.3501), np.float64(25.7494), np.float64(20.0942), np.float64(24.5253), np.float64(17.4807), np.float64(17.6947), np.float64(19.9594), np.float64(20.6254)]

def compare_models(a, b, alpha=0.05):
    a = np.array(a)
    b = np.array(b)

    mean_a, std_a = np.mean(a), np.std(a, ddof=1)
    mean_b, std_b = np.mean(b), np.std(b, ddof=1)

    # Welch's t-test (equal_var=False est recommandé en ML/Data Science)
    t_stat, p_val = stats.ttest_rel(a, b)

    print("=" * 60)
    print("🧪 RÉSULTATS DE LA COMPARAISON STATISTIQUE")
    print("=" * 60)
    print(f"Modèle A : Moyenne = {mean_a:.8f} | Écart-type = {std_a:.8f}")
    print(f"Modèle B : Moyenne = {mean_b:.8f} | Écart-type = {std_b:.8f}")
    print("-" * 60)
    print(f"Statistique t (t-stat) : {t_stat:.4f}")
    print(f"p-value                : {p_val:.6e}")
    print("-" * 60)

    # Interprétation du p-value
    if p_val < alpha:
        print(f"✅ STATISTIQUEMENT SIGNIFICATIF (p < {alpha})")
        print(
            "   -> La différence entre les 2 modèles N'EST PAS due au hasard."
        )
        if mean_b < mean_a:
            print(
                "   -> Le Modèle B est significativement MEILLEUR que le Modèle A."
            )
        else:
            print(
                "   -> Le Modèle A est significativement MEILLEUR que le Modèle B."
            )
    else:
        print(f"❌ NON SIGNIFICATIF (p >= {alpha})")
        print(
            "   -> La différence observée peut simplement être due aux variations aléatoires (bruit)."
        )

    print("=" * 60)

    return t_stat, p_val


if __name__ == "__main__":
    t_stat, p_val = compare_models(model_a, model_b)