def save(model, modelname):
    model= grid.best_estimator_
    model.fit(X, y)
    submission = model.predict(oosample)
    df = pd.DataFrame({'PassengerId':test_df.PassengerId, 
                           'Survived':submission})
    print(len(df))
    df.to_csv('Bagging_ensemble.csv',header=True,index=False)
    run_save()

    os.path.join(r'Titanic/submissions/', "{}.pickle".format(modelname))

    with open(r'Titanic/Pickle/', "{}.pickle".format(modelname), 'wb') as f: pickle.dump(model, f)