from crontab import CronTab
import os
from barefoot_winnie.d00_utils.get_project_directory import get_project_directory
from pathlib import Path

home = Path.home()
paths = get_project_directory()
bash_path = "/bin/bash"


class CronJobs:
    def __init__(self, username):
        sys_path = os.environ.get("PATH")
        self.cron = CronTab(user=username,
                            log=os.path.join("tmp", "cron.log"))
        self.cron.env['PATH'] = sys_path

    def training_pipeline_job(self):
        filename = home.joinpath(
            "barefoot_winnie", "src", "pipeline_trigger.sh")
        print(filename)
        job = self.cron.new(
            command=f"{bash_path} {filename} >> /tmp/pipeline.log 2>&1", comment="training_pipeline")
        job.hour.on(2)
        self.cron.write_to_user()

    def view_jobs(self):
        for job in self.cron:
            print(job)

    def view_log(self):
        for d in self.cron.log:
            print(d['pid'] + " - " + d['date'])

    def remove_all_jobs(self):
        self.cron.remove_all()
        self.cron.write()

    def remove_jobs_by_comment(self, job_comment):
        self.cron.remove_all(comment=job_comment)


if __name__ == "__main__":
    import getpass
    username = getpass.getuser()
    cron_jobs = CronJobs(username)
    cron_jobs.remove_all_jobs()
    cron_jobs.remove_jobs_by_comment('pipeline')
    cron_jobs.training_pipeline_job()
    cron_jobs.view_jobs()
