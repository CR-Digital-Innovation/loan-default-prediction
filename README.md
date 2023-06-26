# loan-default-prediction

 This project predicts the default risk by analyzing the loan application data and it classifies the applicant into three categories:

 - **Confirmed Repayer:** Applicant who likely to repay the loan amount. There is very less to no risk involved with this type of applicants.
 - **Probable Defaulter:** Applicant who is probable to default. There is moderate risk involved with this type of applicants.
 - **Confirmed Defaulter:** Applicant who is likely to default. There is high risk involved with this type of applicants.


 
## Installation steps

**Pre-requisites:**
 - Install docker as mentioned [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script).
 - Install docker-compose using following command:
    `sudo apt install docker-compose -y`

**Setup**
 1. Clone the repository using `git clone` and `cd` into it.
 2. Create a `.env` file in `src/backend/` path with required variables. Check `.sample-env` file for the list of required variables.\
 3. Run following command to build and run the containers.
	 `sudo docker-compose up --build`
 4. Run following command to remove the containers and built images.
	 `sudo docker-compose down --rmi all`
5. Visit `http://<your-ip>:80` to browse the application.
6. Upload or paste the CSV data and click on `Predict` to the get the prediction results.
**Note:** *You can refer to docker-compose documentation for more details.*