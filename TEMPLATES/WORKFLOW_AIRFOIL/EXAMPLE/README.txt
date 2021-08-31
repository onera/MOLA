-----------------------BETA-TEST DU WORKFLOW AIRFOIL MOLA-----------------------


0 - Sourcer un fichier d'environnement situé ici (version Dev de MOLA) en
    fonction de la machine utilisée :

    /home/lbernard/MOLA/Dev/ENVIRONMENTS/

    ATTENTION ! des versions récentes des librairies numpy, scipy et matplotlib
    sont exigées. Pour les mettre à jour, il faut exécuter depuis SPIRO ou EOS
    (après sourcer MOLA) la commande suivante :

    pip install --user --upgrade --force-reinstall numpy scipy cycler matplotlib

    ATTENTION ! Le Workflow nécessite une clef publique ssh permettant d'accéder
    à SATOR sans devoir saisir le mot de passe à chaque fois.

    Si vous rencontrez ce problème, alors procédez à rajouter une clef ssh.
    Merci à Antoine Hajczak pour le partage de cette technique :

    Ouvrir le droits de home et .ssh :

    chmod 755 /home/votrenomdutilisateur
    chmod 755 /home/votrenomdutilisateur/.ssh

    Suivre les steps du tutoriel : https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/ , qui sont
    les suivants :
    ssh-keygen
    ssh-copy-id -i ~/.ssh/id_rsa.pub sator
    ssh sator
    ssh-add

    A la fin de ce processus, vous devriez pouvoir vous connecter à sator sans
    saisir le mot de passe à chaque fois.

    ATTENTION ! Avant de poursuivre, vérifier sur votre poste de travail si
    la configuration de l'environnement est correcte. Pour cela, sourcer
    l'environnement MOLA qui correspond à votre machine, et ensuite lancer
    les commandes suivantes:

    python
    import MOLA.WorkflowAirfoil as WF
    WF.checkDependencies()

    Si l'exécution de la commande précédente pose des problèmes/warning, alors
    contactez luis.bernardos_barreda@onera.fr


1 - Commencer par la configuration du script de lancement PolarLauncher.py
    Choisir un airfoil d'1 mètre de corde, orienté clockwise, et i=1 au bord de
    fuite.

    ATTENTION! à cause du bug Cassiopée #6466, on ne peut faire que des
    maillages en 'O', donc le bord de fuite DOIT ETRE POINTU
    (https://elsa-e.onera.fr/issues/6466)

    ATTENTION! à cause du bug Cassiopée #7517, la première cellule adjecente
    à la paroi n'est pas orthogonale et des défauts peuvent apparaître loin
    dans le sillage dans la jonction du 'C'.
    (https://elsa-e.onera.fr/issues/7517)

    ATTENTION! le script MeshingParameters.py écrase les valeurs par défaut
    des paramètres de génération de maillage.

    Tester la génération d'au moins un maillage avec test-mesher.py, et
    évaluer mesh.cgns. Si maillage acceptable, lancer:

    PolarLauncher.py


2 - Utiliser le script PolarMonitor.py pour suivre l'état des calculs.


3 - Une fois tous les calculs terminés, lancer le script PolarBuilder.py pour
    créer un nouveau fichier de polaires.


4 - Evaluer la qualité des polaires en les traçant, à l'aide de PolarPlotter.py
    Toute quantité contenue dans le FlowSolution du fichier CGNS de polaires
    construit avec PolarBuilder.py peut être tracé avec PolarPlotter.py

5 - Eventuellement corriger les polaires en remplaçant les cas mal convergés par
    des valeurs issus des grands angles imposés, à l'aide de PolarCorrector.py
