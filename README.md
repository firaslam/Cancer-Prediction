# A toy breast cancer ML classification model deployed in Flask and Docker

sudo docker build -t breast_cancer_api .
docker run -p 5000:5000 breast_cancer_api
```

## to push image to docker hub   

  docker tag local-image:tagname new-repo:tagname
  docker push new-repo:tagname


  ------
  to delete deployment

  kubectl get rs --all-namespaces
  kubectl delete deployment  model-deployment --namespace=default 

  -------
  # Deploy Deployment

kubectl apply -f breast_cancer_api _deployment.yaml

  # Deploy services

kubectl apply -f prediction-service.yaml

  # Get all 

kubectl get all

# kubectl delete

kubectl delete --all services



