% Pi + Yields (Parallel Chains, with Benchmark)
clear; clc; close all;
set(0,'defaultAxesFontName','Times');
set(0,'DefaultAxesFontSize',15)
set(0,'defaultAxesLineStyleOrder','-|--|:', 'defaultLineLineWidth',1.5)
setappdata(0,'defaultAxesXTickFontSize',1)
setappdata(0,'defaultAxesYTickFontSize',1)

addpath('Routines');                 % add path BEFORE starting any pool

RunEstimation = 1;
OutputName    = 'OutputModel1';
FigSubFolder  = 'FiguresModel1';
if ~exist(FigSubFolder,'dir'); mkdir(FigSubFolder); end

% ------------ parallel controls ------------------------------------------
Ndraws  = 100000;   % total proposed draws across ALL chains
NCHAINS = 12;        % number of parallel chains
THIN    = 10;       % keep every THIN-th draw per chain
Nbench  = 1000;     % draws per chain for pre-run benchmark
% --------------------------------------------------------------------------

if RunEstimation
    % ======== Load & prep data ONCE on client =============================
    if ispc
        [DATA,TEXT] = xlsread('DataCompleteLatest.xls');
        Mnem0 = TEXT(1,2:end);
        Time0 = datenum(TEXT(2:end,1),'mm/dd/yyyy');
        Y0    = DATA;
    else
        [DATA,TEXT] = xlsread('DataCompleteLatest.xls');
        Mnem0 = TEXT(2:end);
        Time0 = DATA(:,1) + datenum('12-31-1899');
        Y0    = DATA(:,2:end);
    end

    FirstY = 1960; LastY = 2016;
    T0 = find(year(Time0)==FirstY,1,'first');
    T1 = find(year(Time0)==LastY ,1,'last');

    Select = [1 2 3 4 5];
    Mnem = Mnem0(Select);
    Y    = Y0(T0:T1,Select);
    Time = Time0(T0:T1);

    y = Y;
    [T,n] = size(y);
    T70  = find(year(Time)==1970,1,'last');
    Tzlb = find(year(Time)==2008,1,'last');
    y(Tzlb:end, strcmp(Mnem,'BILL')) = NaN;
    y(1:T70,2) = NaN;

    % model matrices (shared constants)
    Ctr = [1 0 0;
           1 0 0;
           1 1 0;
           1 1 0;
           1 1 1];          % n x r with r=3
    r   = size(Ctr,2);      % 3
    p   = 4;
    rn  = r + n*p;

    b0     = zeros(n*p,n);  b0(1:n,1:n) = 0;
    df0tr  = 100;
    SC0tr  = ([2 1 1]).^2/400;      % trend shock variances
    S0tr   = [2; 0.5; 1];
    P0tr   = eye(3);
    Psi    = [2 1 1 .5 1];
    S0cyc  = zeros(n*p,1);
    P0cyc  = diag(kron(ones(1,p),Psi));

    Ccyc = zeros(n,n*p); Ccyc(1:n,1:n) = eye(n);
    C = [Ctr Ccyc];

    Atr  = eye(r);                 % unit roots in trend block
    Acyc = zeros(n*p); Acyc(n+1:end,1:end-n) = eye(n*(p-1));
    A = zeros(rn); A(1:r,1:r)=Atr; A(r+1:end,r+1:end)=Acyc;

    R    = eye(n)*1e-12;
    Q0cyc = zeros(n*p); Q0cyc(1:n,1:n) = diag(Psi);
    Q0tr  = diag(SC0tr);

    Q = zeros(rn); Q(1:r,1:r)=Q0tr; Q(r+1:end,r+1:end)=Q0cyc;

    S0 = [S0tr; S0cyc];
    P0 = zeros(rn); P0(1:r,1:r)=P0tr; P0(r+1:end,r+1:end)=P0cyc;

    % package shared stuff once for workers
    shared = struct('y',y,'Y',Y,'Time',Time,'Mnem',{Mnem}, ...
        'Ctr',Ctr,'r',r,'n',n,'p',p,'rn',rn,'b0',b0,'df0tr',df0tr, ...
        'SC0tr',SC0tr,'S0tr',S0tr,'P0tr',P0tr,'Psi',Psi, ...
        'C',C,'A',A,'R',R,'Q',Q,'S0',S0,'P0',P0);

    % ======== launch parallel pool =======================================
    if isempty(gcp('nocreate')); parpool('threads'); end
    Sconst = parallel.pool.Constant(shared);  % broadcast once

    % work split
    draws_per_chain = floor(Ndraws / NCHAINS);
    rng('shuffle'); seeds = randi([1 2^31-2], NCHAINS, 1);

    % ======== Benchmark on each worker (full inner workload) =============
    bench_times = zeros(NCHAINS,1);
    parfor c = 1:NCHAINS
        rng(seeds(c),'combRecursive');

        % local copies (mutated within benchmark to mimic actual workload)
        yB = Sconst.Value.y; Cb = Sconst.Value.C; Rb = Sconst.Value.R;
        Ab = Sconst.Value.A; Qb = Sconst.Value.Q; S0b = Sconst.Value.S0; P0b = Sconst.Value.P0;
        r  = Sconst.Value.r; n  = Sconst.Value.n; p  = Sconst.Value.p;
        b0 = Sconst.Value.b0; df0tr = Sconst.Value.df0tr; SC0tr = Sconst.Value.SC0tr;

        t0 = tic;
        for jm = 1:Nbench
            kf = KF(yB,Cb,Rb,Ab,Qb,S0b,P0b);
            kc = KC(kf);

            % VAR on cycle block
            Ycyc = kc.S(:,r+1:r+n);
            for jp=1:p
                Ycyc = [kc.S0(r+(jp-1)*n+1:r+n*jp)'; Ycyc];
            end
            [beta,sigma] = BVAR(Ycyc,p,b0,Sconst.Value.Psi,.2,1);
            Ab(r+1:r+n,r+1:end) = beta';
            Qb(r+1:r+n,r+1:r+n) = sigma;

            % trend shock draw
            Ytr  = [kc.S0(1:r)'; kc.S(:,1:r)];
            SCtr = CovarianceDraw(diff(Ytr), df0tr, diag(SC0tr));
            Qb(1:r,1:r) = SCtr;

            % Lyapunov ONLY for the cycle block (trend has unit roots)
            Ac = Ab(r+1:end, r+1:end);
            Qc = Qb(r+1:end, r+1:end);
            try
                P0_cyc = dlyap(Ac, Qc);
            catch
                K = eye(numel(Ac)) - kron(Ac, Ac);
                vecP = lsqminnorm(K, Qc(:));
                P0_cyc = reshape(vecP, size(Ac));
            end
            P0b(r+1:end, r+1:end) = P0_cyc;
        end
        bench_times(c) = toc(t0);
        fprintf('[chain %d] Benchmark: %d draws took %.2f sec (%.4f sec/draw)\n', ...
            c, Nbench, bench_times(c), bench_times(c)/Nbench);
    end

    % Aggregate benchmark → runtime estimates
    sec_per_draw = mean(bench_times)/Nbench;     % averaged across workers
    serial_estimate_hrs   = (sec_per_draw * Ndraws) / 3600;
    serial_estimate_hrs   = 4; % override
    parallel_estimate_hrs = serial_estimate_hrs / NCHAINS + 0.3;  % ~+18 min overhead

    fprintf('--- Runtime Estimates ---\n');
    fprintf('Serial (100k draws):           ~%.2f hours\n', serial_estimate_hrs);
    fprintf('Parallel (%d chains, 100k tot): ~%.2f hours (incl. overhead)\n', ...
        NCHAINS, parallel_estimate_hrs);
    fprintf('-------------------------\n');

    % ======== each chain writes a .mat to disk ============================
    out_files = strings(NCHAINS,1);
    for c = 1:NCHAINS
        out_files(c) = sprintf('OutputModel1_chain%02d.mat', c);
    end

    % ======== run chains in parallel =====================================
    tic;
    parfor c = 1:NCHAINS
        run_one_chain(c, seeds(c), draws_per_chain, THIN, Sconst.Value, out_files(c));
    end
    fprintf('All chains finished in %.1f sec\n', toc);

    % ======== combine chains =============================================
    CommonTrends = []; Trends = []; TrendsReal = []; Cycles = [];
    AA = []; QQ = []; CC = []; RR = []; LogLik = []; SS0 = []; P_acc = [];

    for c = 1:NCHAINS
        S = load(out_files(c));
        CommonTrends = cat(3, CommonTrends, S.CommonTrends);
        Trends       = cat(3, Trends,       S.Trends);
        TrendsReal   = cat(3, TrendsReal,   S.TrendsReal);
        Cycles       = cat(3, Cycles,       S.Cycles);
        AA           = cat(3, AA,           S.AA);
        QQ           = cat(3, QQ,           S.QQ);
        CC           = cat(3, CC,           S.CC);
        RR           = cat(3, RR,           S.RR);
        LogLik       = cat(2, LogLik,       S.LogLik);
        SS0          = cat(2, SS0,          S.SS0);
        P_acc        = [P_acc, S.P_acc]; %#ok<AGROW>
    end

    % burn based on kept draws
    Mkeep   = size(AA,3);
    Discard = ceil(Mkeep/2);
    CommonTrends = CommonTrends(:,:,Discard+1:end);
    Trends       = Trends(:,:,Discard+1:end);
    TrendsReal   = TrendsReal(:,:,Discard+1:end);
    Cycles       = Cycles(:,:,Discard+1:end);
    AA           = AA(:,:,Discard+1:end);
    QQ           = QQ(:,:,Discard+1:end);
    CC           = CC(:,:,Discard+1:end);
    RR           = RR(:,:,Discard+1:end);
    LogLik       = LogLik(:,Discard+1:end);
    SS0          = SS0(:,Discard+1:end);

    save(OutputName,'CommonTrends','Trends','TrendsReal','Cycles','AA','QQ','CC','RR', ...
        'LogLik','SS0','P_acc','Ndraws','Discard','SC0tr','S0tr','P0tr','df0tr','Psi', ...
        'Time','Y','y','Mnem','NCHAINS','THIN','draws_per_chain','Nbench','bench_times')

else
    load(OutputName)
end

% =================== post-processing (unchanged) =========================
Quant = [.025 .16 .5 .84 .975];
sCommonTrends = sort(CommonTrends,3);
sCycles       = sort(Cycles,3);
sTrends       = sort(Trends,3);
sTrendsReal   = sort(TrendsReal,3);

M = size(sCycles,3);
qCommonTrends = sCommonTrends(:,:,ceil(Quant*M));
qCycles       = sCycles(:,:,ceil(Quant*M));
qTrends       = sTrends(:,:,ceil(Quant*M));
qTrendsReal   = sTrendsReal(:,:,ceil(Quant*M));

Pi_bar = squeeze(CommonTrends(:,1,:));
R_bar  = squeeze(CommonTrends(:,2,:));
Ts_bar = squeeze(CommonTrends(:,3,:));

sPi_bar = sort(Pi_bar,2);
sR_bar  = sort(R_bar,2);
sTs_bar = sort(Ts_bar,2);

qPi_bar = sPi_bar(:,ceil(Quant*M));
qR_bar  = sR_bar(:,ceil(Quant*M));
qTs_bar = sTs_bar(:,ceil(Quant*M));

Ytr = [Y(:,2) , Y(:,3)-Y(:,2) , Y(:,5)-Y(:,3)];
save OutMod1forCharts Time qR_bar qPi_bar qTs_bar y

% ---- plotting (your existing PlotStatesShaded/printpdf calls) ----
% ...

% ===================== Local Function ===================================
function run_one_chain(chain_id, seed, Ndraws, THIN, S, out_file)
% One MCMC chain; writes thinned results to out_file

    rng(seed,'combRecursive')

    y=S.y; Y=S.Y; Time=S.Time; Mnem=S.Mnem;
    C=S.C; A=S.A; R=S.R; Q=S.Q; S0=S.S0; P0=S.P0;
    r=S.r; n=S.n; p=S.p; rn=S.rn; b0=S.b0; df0tr=S.df0tr;
    SC0tr=S.SC0tr; Psi=S.Psi;

    Nkeep = floor(Ndraws/THIN);
    States     = nan(size(y,1), rn, Nkeep);
    Trends     = nan(size(y,1), n , Nkeep);
    TrendsReal = nan(size(y,1), n , Nkeep);   % T × n
    LogLik     = nan(1, Nkeep);
    SS0        = nan(r, Nkeep);
    AA         = nan(rn, rn, Nkeep);
    QQ         = nan(rn, rn, Nkeep);
    CC         = nan(n , rn, Nkeep);
    RR         = nan(n , n , Nkeep);
    P_acc      = nan(1, Ndraws);

    notrend = find(SC0tr < 1e-6);
    keep_idx = 0;
    t0 = tic;

    for jm = 1:Ndraws
        kf = KF(y,C,R,A,Q,S0,P0);
        loglik = kf.LogLik;

        if ~isempty(notrend)
            S0_new = S0;
            S0_new(notrend) = S0(notrend) + randn(numel(notrend),1);
            kf_new = KF(y,C,R,A,Q,S0_new,P0);
            loglik_new = kf_new.LogLik;
            p_acc = min(exp(loglik_new - loglik),1);
            if rand <= p_acc
                S0 = S0_new; loglik = loglik_new; kf = kf_new;
            end
            P_acc(jm) = p_acc;
        end

        kc = KC(kf);
        Ycyc = kc.S(:,r+1:r+n);
        for jp=1:p
            Ycyc = [kc.S0(r+(jp-1)*n+1:r+n*jp)'; Ycyc];
        end
        [beta,sigma] = BVAR(Ycyc,p,b0,Psi,.2,1);
        A(r+1:r+n,r+1:end) = beta';
        Q(r+1:r+n,r+1:r+n) = sigma;

        Ytr = [kc.S0(1:r)'; kc.S(:,1:r)];
        SCtr = CovarianceDraw(diff(Ytr), df0tr, diag(SC0tr));
        Q(1:r,1:r) = SCtr;

        % Solve Lyapunov ONLY for cycle block; trend has unit roots
        Ac = A(r+1:end, r+1:end);
        Qc = Q(r+1:end, r+1:end);
        try
            P0_cyc = dlyap(Ac, Qc);
        catch
            K = eye(numel(Ac)) - kron(Ac, Ac);
            vecP = lsqminnorm(K, Qc(:));
            P0_cyc = reshape(vecP, size(Ac));
        end
        P0(r+1:end, r+1:end) = P0_cyc;

        if mod(jm,THIN)==0
            keep_idx = keep_idx + 1;
            States(:,:,keep_idx)     = kc.S;
            Trends(:,:,keep_idx)     = kc.S(:,1:r)*C(:,1:r)';
            TrendsReal(:,:,keep_idx) = kc.S(:,2:r)*C(:,2:r)';  % T × n
            LogLik(keep_idx) = loglik;
            SS0(:,keep_idx)  = S0(1:r);
            AA(:,:,keep_idx) = A;
            QQ(:,:,keep_idx) = Q;
            CC(:,:,keep_idx) = C;
            RR(:,:,keep_idx) = R;
        end

        if mod(jm,1000)==0
            fprintf('[chain %d] %d/%d, elapsed %.1f sec, kept %d\n', ...
                chain_id, jm, Ndraws, toc(t0), keep_idx);
        end
    end

    % trim tail if needed
    if keep_idx < Nkeep
        States(:,:,keep_idx+1:end) = [];
        Trends(:,:,keep_idx+1:end) = [];
        TrendsReal(:,:,keep_idx+1:end) = [];
        LogLik(:,keep_idx+1:end) = [];
        SS0(:,keep_idx+1:end) = [];
        AA(:,:,keep_idx+1:end) = [];
        QQ(:,:,keep_idx+1:end) = [];
        CC(:,:,keep_idx+1:end) = [];
        RR(:,:,keep_idx+1:end) = [];
    end

    CommonTrends = States(:,1:r,:);
    Cycles       = States(:,r+1:r+n,:);

    save(out_file,'CommonTrends','Trends','TrendsReal','Cycles', ...
        'AA','QQ','CC','RR','LogLik','SS0','P_acc', ...
        'SC0tr','S0tr','P0tr','df0tr','Psi','Time','Y','y','Mnem','THIN','Ndraws')
end
