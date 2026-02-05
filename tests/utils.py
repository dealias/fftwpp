import sys
import os
import collections
import subprocess
from math import ceil, floor, isclose
import time

def ceilpow2(n):
  n-=1
  n |= n >> 1
  n |= n >> 2
  n |= n >> 4
  n |= n >> 8
  n |= n >> 16
  n |= n >> 32
  return n+1

def ceilquotient(a,b):
  return -(a//-b)

def nextfftsize(m):
  N=ceilpow2(m)
  if m == N:
    return N
  ni=1
  while ni < N:
    nj=ni
    while nj < N:
      nk=nj
      while nk < N:
        N=min(N,nk*ceilpow2(ceilquotient(m,nk)))
        nk*=3
      nj*=5
    ni*=7
  return N

def usecmd(cmd):
  try:
    t0=time.time()
    vp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = vp.communicate()
    prc = vp.returncode
    output = out.rstrip().decode() if out else ""
    if prc == 0:
        return output
    else:
        return output

  except Exception as e:
    return f"Error running command: {e}"

def readable_list(seq):
    """Return a grammatically correct human readable string (with an Oxford comma)."""
    # Ref: https://stackoverflow.com/a/53981846/
    seq = [str(s) for s in seq]
    if len(seq) < 3:
        return ' and '.join(seq)
    return ', '.join(seq[:-1]) + ', and ' + seq[-1]

def supports_ansi():
    if not sys.stdout.isatty():
        return False
    term = os.environ.get('TERM', '')
    if 'xterm' in term or 'screen' in term or 'linux' in term:
        return True
    return False

def seconds_to_readable_time(seconds):
    if seconds < 0:
        raise ValueError("Seconds must be non-negative.")

    total_minutes = seconds / 60
    total_hours = seconds / 3600
    total_days = seconds / 86400

    if total_days >= 1:
        # Round to nearest hour and convert to days + hours
        rounded_hours = round(seconds / 3600)
        days = rounded_hours // 24
        hours = rounded_hours % 24

        parts = [f"{days} day{'s' if days != 1 else ''}"]
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        return ", ".join(parts)

    elif total_hours >= 1:
        # Round to nearest minute and convert to hours + minutes
        rounded_minutes = round(seconds / 60)
        hours = rounded_minutes // 60
        minutes = rounded_minutes % 60

        parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        return ", ".join(parts)

    elif total_minutes > 5:
        rounded_minutes = round(total_minutes)
        return f"{rounded_minutes} minute{'s' if rounded_minutes != 1 else ''}"

    else:
        # For ≤ 5 minutes, show minutes + seconds (to 2 decimals)
        minutes = floor(seconds / 60)
        remaining_seconds = seconds - minutes * 60

        parts = []
        if minutes == 1:
            parts.append("1 minute")
        elif minutes > 0:
            parts.append(f"{minutes} minutes")

        if isclose(remaining_seconds, 1.0, abs_tol=1e-9):
            parts.append("1 second")
        elif remaining_seconds > 0:
            parts.append(f"{round(remaining_seconds):d} seconds")

        return ", ".join(parts) if parts else "0 seconds"

def send_email(subject:str,content:str):
    import smtplib
    from email.message import EmailMessage
    from dotenv import load_dotenv
    # Load credentials and settings from .env
    load_dotenv(override=True)
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER   = os.getenv("SMTP_USER")
    SMTP_PASS   = os.getenv("SMTP_PASS")
    TO_EMAIL    = os.getenv("TO_EMAIL")

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = TO_EMAIL
    msg["Subject"] = subject
    msg.set_content(content)
    try:
      with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
          server.ehlo()            # Identify to server
          server.starttls()        # Upgrade to secure connection
          server.ehlo()
          server.login(SMTP_USER, SMTP_PASS)
          server.send_message(msg)
    except Exception as e:
      print("SMTP error:", e)


class Progress:
  def __init__(self):
    self.n = 0
    self.mean = 0.0
    self.estimatedtime=float('inf')
    self.total_tests=0
    self.untimed_tests=0
    self.min_tests_for_time_estimate=5
    self.time_for_estimate=2

  def update(self, x):
    self.n += 1
    delta = x - self.mean
    self.mean += delta / self.n

  def report(self):
    approximate_time=""
    if self.n*self.mean > self.time_for_estimate and self.n > self.min_tests_for_time_estimate:
      self.estimatedtime=ceil(min(self.estimatedtime,(self.total_tests-self.n)*self.mean))
      approximate_time=f" (approximately {seconds_to_readable_time(self.estimatedtime)} remaining)"

    print(f"\rProgress: {self.n+self.untimed_tests}/{self.total_tests}{approximate_time}.\033[K", end="")

    if self.total_tests-self.n <= self.untimed_tests:
      print("\r\033[K",end="")

class ParameterCollection():
# Holds a collection of Params for a given program

  def __init__(self, vals=[]):
    # fillParameters: A funtion that returns a list of Parameters.
    #
    # optArgs: Optional arguments for fillParameters. If None, then
    # fillParameters is called without arguments.
    if type(vals) == list:
      for value in vals:
        if type(value) != Parameters:
          exit("fillOptValues must return a list of Parameters.")
      self.vals=vals
    else:
      exit("fillOptValues must return a list of Parameters.")

  def __eq__(self, other):
    if isinstance(other, ParameterCollection):
      return collections.Counter(self.vals) == collections.Counter(other.vals)
    return False

  # sortX() sorts self.val by the value of X
  def sortL(self, reverse=False):
    def byL(v):
      return v.L
    self.vals.sort(reverse=reverse,key=byL)

  def sortM(self, reverse=False):
    def f(v):
      return v.M
    self.vals.sort(reverse=reverse,key=f)

  def sortm(self, reverse=False):
    def f(v):
      return v.m
    self.vals.sort(reverse=reverse,key=f)

  def sortp(self, reverse=False):
    def f(v):
      return v.p
    self.vals.sort(reverse=reverse,key=f)

  def sortq(self, reverse=False):
    def f(v):
      return v.q
    self.vals.sort(reverse=reverse,key=f)

  def sortC(self, reverse=False):
    def f(v):
      return v.C
    self.vals.sort(reverse=reverse,key=f)

  def sortS(self, reverse=False):
    def f(v):
      return v.S
    self.vals.sort(reverse=reverse,key=f)

  def sortD(self, reverse=False):
    def f(v):
      return v.D
    self.vals.sort(reverse=reverse,key=f)

  def sortI(self, reverse=False):
    def f(v):
      return v.I
    self.vals.sort(reverse=reverse,key=f)

  def sortt(self, reverse=False):
    def f(v):
      return v.t
    self.vals.sort(reverse=reverse,key=f)

  # Returns a list of L values (respecting the order of self.val)
  def L(self):
    values=[]
    for v in self.vals:
      values.append(v.L)
    return values

  # Returns a list of M values (respecting the order of self.val)
  def M(self):
    values=[]
    for v in self.vals:
      values.append(v.M)
    return values

  # Returns a list of m values (respecting the order of self.val)
  def m(self):
    values=[]
    for v in self.vals:
      values.append(v.m)
    return values

  # Returns a list of p values (respecting the order of self.val)
  def p(self):
    values=[]
    for v in self.vals:
      values.append(v.p)
    return values

  # Returns a list of q values (respecting the order of self.val)
  def q(self):
    values=[]
    for v in self.vals:
      values.append(v.q)
    return values

  # Returns a list of C values (respecting the order of self.val)
  def C(self):
    values=[]
    for v in self.vals:
      values.append(v.C)
    return values

  # Returns a list of S values (respecting the order of self.val)
  def S(self):
    values=[]
    for v in self.vals:
      values.append(v.S)
    return values

  # Returns a list of D values (respecting the order of self.val)
  def D(self):
    values=[]
    for v in self.vals:
      values.append(v.D)
    return values

  # Returns a list of I values (respecting the order of self.val)
  def I(self):
    values=[]
    for v in self.vals:
      values.append(v.I)
    return values

  # Returns a list of t values (respecting the order of self.val)
  def t(self):
    values=[]
    for v in self.vals:
      values.append(v.t)
    return values

  # Returns a dictionary of lists of values (respecting the order of self.val)
  def allParams(self):
    params={}
    params["L"]=self.L()
    params["M"]=self.M()
    params["m"]=self.m()
    params["p"]=self.p()
    params["q"]=self.q()
    params["C"]=self.C()
    params["S"]=self.S()
    params["D"]=self.D()
    params["I"]=self.I()
    params["t"]=self.t()
    return params

class Parameters():
# A collection of parameters for a hybrid dealiased convolution
# Two Parameters objects are equal if all of their parameters OTHER THAN t are equal.
  def __init__(self,L,M,m,p,q,C,S,D,I,t=-1):
    self.L=int(L)
    self.M=int(M)
    self.m=int(m)
    self.p=int(p)
    self.q=int(q)
    self.C=int(C)
    self.S=int(S)
    self.D=int(D)
    self.I=int(I)
    self.t=float(t)

  def __hash__(self):
    return hash((self.L, self.M, self.m, self.p, self.q, self.C, self.S, self.D, self.I))

  def __eq__(self, other):
    if isinstance(other, Parameters):
      if self.L != other.L:
        return False
      if self.M != other.M:
        return False
      if self.m != other.m:
        return False
      if self.p != other.p:
        return False
      if self.q != other.q:
        return False
      if self.C != other.C:
        return False
      if self.S != other.S:
        return False
      if self.D != other.D:
        return False
      if self.I != other.I:
        return False
      return True
    return False
