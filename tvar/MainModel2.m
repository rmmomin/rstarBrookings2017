% Pi + Yields + BAA  (Parallel Chains, benchmark, thinning, post-processing)
clear; clc; close all;
set(0,'defaultAxesFontName','Times');
set(0,'DefaultAxesFontSize',15)
set(0,'defaultAxesLineStyleOrder','-|--|:', 'defaultLineLineWidth',1.5)
setappdata(0,'defaultAxesXTickFontSize',1)
setappdata(0,'defaultAxesYTickFontSize',1)

addpath('Routines');                 % ensure Routines is on path

RunEstimation = 1;
OutputName    = 'OutputModel2';
FigSubFolder  = 'FiguresModel2';
if ~exist(FigSubFolder,'dir'); mkdir(FigSubFolder); end

% ------------ controls ----------------------------------------------------
Ndraws     = 100000;  % total draws across ALL chains
NCHAINS    = 8;       % match to physical cores
THIN       = 10;      % keep every THIN-th draw
Nbench     = 1000;    % per-chain benchmark draws
SAVE_LEVEL = 'lite';  % 'lite' or 'full'
% -------------------------------------------------------------------------

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

    % Presample diagnostics (kept from original Model2)
    T0pre = find(year(Time0)==1954,1,'first');
    T1pre = find(year(Time0)==1959,1,'last');
    if ~isempty(T0pre) && ~isempty(T1pre)
        disp('Avg. and std in the presample: 1954-1959'); disp(Mnem0);
        disp(mean(Y0(T0pre:T1pre,:))); disp(std(Y0(T0pre:T1pre,:)));
    end

    FirstY = 1960; LastY = 2016;
    T0 = find(year(Time0)==FirstY,1,'first');
    T1 = find(year(Time0)==LastY ,1,'last');

    % Variables (Model2 adds BAA)
    Select = [1 2 3 4 5 7];
    Mnem = Mnem0(Select);
    Y    = Y0(T0:T1,Select);
    Time = Time0(T0:T1);

    y = Y;
    [T,n] = size(y);
    T70  = find(year(Time)==1970,1,'last');
    Tzlb = find(year(Time)==2008,1,'last');
    y(Tzlb:end, strcmp(Mnem,'BILL')) = NaN;  % ZLB missing bills
    y(1:T70,2) = NaN;                        % early missing exp. inflation

    % ---------- Model2 identification (4 trends: pi, r, Ts, Cy) ----------
    % rows: [pi, Epi, bill, Eshort, long, BAA]
    Ctr = [ 1 0 0  0 ;   % pi
            1 0 0  0 ;   % Epi
            1 1 0 -1 ;   % 3m bill = r - Cy
            1 1 0 -1 ;   % E short = r - Cy
            1 1 1 -1 ;   % long = r + Ts - Cy
            1 1 1  0 ];  % BAA = r + Ts
    r   = size(Ctr,2);        % 4 trends
    p   = 4;                  % VAR lags for cycle
    rn  = r + n*p;

    % Priors / initial states (as in original Model2)
    b0     = zeros(n*p,n);  b0(1:n,1:n) = 0;
    df0tr  = 100;
    SC0tr  = ([2, 1/sqrt(2), 1, 1/sqrt(2)]).^2/400;  % trend shock variances
    S0tr   = [2; 1.5; 1; 1];
    P0tr   = eye(r);

    % cycle innovation scales per variable (pi, Epi, bill, Eshort, long, BAA)
    Psi    = [2 1 1 .5 1 1];

    S0cyc  = zeros(n*p,1);
    P0cyc  = diag(kron(ones(1,p),Psi));

    % State-space blocks
    Ccyc = zeros(n,n*p); Ccyc(1:n,1:n) = eye(n);
    C = [Ctr Ccyc];

    Atr  = eye(r);
    Acyc = zeros(n*p); Acyc(n+1:end,1:end-n) = eye(n*(p-1));
    A = zeros(rn); A(1:r,1:r)=Atr; A(r+1:end,r+1:end)=Acyc;

    R    = eye(n)*1e-12;
    Q0cyc = zeros(n*p); Q0cyc(1:n,1:n) = diag(Psi);
    Q0tr  = diag(SC0tr);
    Q = zeros(rn); Q(1:r,1:r)=Q0tr; Q(r+1:end,r+1:end)=Q0cyc;

    S0 = [S0tr; S0cyc];
    P0 = zeros(rn); P0(1:r,1:r)=P0tr; P0(r+1:end,r+1:end)=P0cyc;

    % ======== launch PROCESS pool ========================================
    ppool = gcp('nocreate'); if ~isempty(ppool), delete(ppool); end
    parpool('local',NCHAINS);

    % Package once for workers
    shared = struct('y',y,'Y',Y,'Time',Time,'Mnem',{Mnem}, ...
        'Ctr',Ctr,'r',r,'n',n,'p',p,'rn',rn,'b0',b0,'df0tr',df0tr, ...
        'SC0tr',SC0tr,'S0tr',S0tr,'P0tr',P0tr,'Psi',Psi, ...
        'C',C,'A',A,'R',R,'Q',Q,'S0',S0,'P0',P0,'SAVE_LEVEL',SAVE_LEVEL);
    Sconst = parallel.pool.Constant(shared);

    % Work split + seeds
    draws_per_chain = floor(Ndraws / NCHAINS);
    rng('shuffle'); seeds = randi([1 2^31-2], NCHAINS, 1);

    % ======== Benchmark ===================================================
    bench_times = zeros(NCHAINS,1);
    parfor c = 1:NCHAINS
        rng(seeds(c),'combRecursive');
        yB=Sconst.Value.y; CB=Sconst.Value.C; RB=Sconst.Value.R;
        AB=Sconst.Value.A; QB=Sconst.Value.Q; S0b=Sconst.Value.S0; P0b=Sconst.Value.P0;
        r = Sconst.Value.r; n = Sconst.Value.n; p = Sconst.Value.p;
        b0=Sconst.Value.b0; df0tr=Sconst.Value.df0tr; SC0tr=Sconst.Value.SC0tr; Psi=Sconst.Value.Psi;

        t0=tic;
        for jm=1:Nbench
            kf = KF(yB,CB,RB,AB,QB,S0b,P0b);
            kc = KC(kf);

            Ycyc = kc.S(:,r+1:r+n);
            for jp=1:p
                Ycyc = [kc.S0(r+(jp-1)*n+1:r+n*jp)'; Ycyc];
            end
            [beta,sigma] = BVAR(Ycyc,p,b0,Psi,.2,1);
            AB(r+1:r+n,r+1:end) = beta';
            QB(r+1:r+n,r+1:r+n) = sigma;

            Ytr  = [kc.S0(1:r)'; kc.S(:,1:r)];
            SCtr = CovarianceDraw(diff(Ytr), df0tr, diag(SC0tr));
            QB(1:r,1:r) = SCtr;

            % steady-state P0 for cycle (robust dlyap)
            Ac = AB(r+1:end, r+1:end);
            Qc = QB(r+1:end, r+1:end);
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
    sec_per_draw = mean(bench_times)/Nbench;
    fprintf('--- Runtime Estimates ---\n');
    fprintf('Serial (100k draws):           ~%.2f hours\n', (sec_per_draw*Ndraws)/3600);
    fprintf('Parallel (%d chains):          ~%.2f hours (incl. overhead)\n', ...
        NCHAINS, (sec_per_draw*Ndraws)/3600/NCHAINS + 0.3);
    fprintf('-------------------------\n');

    % ======== per-chain output paths =====================================
    ChainsOutDir = 'chains_out_model2';
    if ~exist(ChainsOutDir,'dir'), mkdir(ChainsOutDir); end
    out_files = strings(NCHAINS,1);
    for c = 1:NCHAINS
        out_files(c) = fullfile(ChainsOutDir, sprintf('OutputModel2_chain%02d.mat', c));
    end

    % ======== run chains in parallel =====================================
    tic;
    parfor c = 1:NCHAINS
        run_one_chain_M2(c, seeds(c), draws_per_chain, THIN, Sconst.Value, out_files(c));
    end
    fprintf('All chains finished in %.1f sec\n', toc);

    % ======== combine chains =============================================
    CommonTrends = []; Trends = []; TrendsReal = []; Cycles = [];
    if strcmpi(SAVE_LEVEL,'full')
        AA = []; QQ = []; CC = []; RR = []; LogLik = []; SS0 = []; P_acc = [];
    end
    for c = 1:NCHAINS
        S = load(out_files(c));
        CommonTrends = cat(3, CommonTrends, S.CommonTrends);
        Trends       = cat(3, Trends,       S.Trends);
        TrendsReal   = cat(3, TrendsReal,   S.TrendsReal);
        Cycles       = cat(3, Cycles,       S.Cycles);
        if strcmpi(SAVE_LEVEL,'full')
            if isfield(S,'AA'),     AA = cat(3, AA, S.AA); end
            if isfield(S,'QQ'),     QQ = cat(3, QQ, S.QQ); end
            if isfield(S,'CC'),     CC = cat(3, CC, S.CC); end
            if isfield(S,'RR'),     RR = cat(3, RR, S.RR); end
            if isfield(S,'LogLik'), LogLik = cat(2, LogLik, S.LogLik); end
            if isfield(S,'SS0'),    SS0 = cat(2, SS0, S.SS0); end
            if isfield(S,'P_acc'),  P_acc = [exist('P_acc','var')*P_acc, S.P_acc]; %#ok<AGROW>
        end
    end

    Mkeep   = size(CommonTrends,3);
    Discard = ceil(Mkeep/2);
    CommonTrends = CommonTrends(:,:,Discard+1:end);
    Trends       = Trends(:,:,Discard+1:end);
    TrendsReal   = TrendsReal(:,:,Discard+1:end);
    Cycles       = Cycles(:,:,Discard+1:end);

    if strcmpi(SAVE_LEVEL,'full')
        if exist('AA','var'), AA = AA(:,:,Discard+1:end); end
        if exist('QQ','var'), QQ = QQ(:,:,Discard+1:end); end
        if exist('CC','var'), CC = CC(:,:,Discard+1:end); end
        if exist('RR','var'), RR = RR(:,:,Discard+1:end); end
        if exist('LogLik','var'), LogLik = LogLik(:,Discard+1:end); end
        if exist('SS0','var'), SS0 = SS0(:,Discard+1:end); end
    end

    % ---- final save ------------------------------------------------------
    if strcmpi(SAVE_LEVEL,'lite')
        save(OutputName,'CommonTrends','Trends','TrendsReal','Cycles', ...
            'Ndraws','Discard','SC0tr','S0tr','P0tr','df0tr','Psi', ...
            'Time','Y','y','Mnem','NCHAINS','THIN','draws_per_chain','Nbench','bench_times','-v7.3');
    else
        save(OutputName,'CommonTrends','Trends','TrendsReal','Cycles', ...
            'AA','QQ','CC','RR','LogLik','SS0','P_acc', ...
            'Ndraws','Discard','SC0tr','S0tr','P0tr','df0tr','Psi', ...
            'Time','Y','y','Mnem','NCHAINS','THIN','draws_per_chain','Nbench','bench_times','-v7.3');
    end
else
    load(OutputName)
end

%% === POST-PROCESS: quantiles, CSV, and figures ==========================
if ~exist('CommonTrends','var'), load(OutputName); end
if ~exist('FigSubFolder','var') || ~exist(FigSubFolder,'dir')
    FigSubFolder = 'FiguresModel2'; mkdir(FigSubFolder);
end

Quant = [.025 .16 .50 .84 .975];

sCommonTrends = sort(CommonTrends,3);
sCycles       = sort(Cycles,3);
sTrends       = sort(Trends,3);
sTrendsReal   = sort(TrendsReal,3);

M = size(sCycles,3);
qCommonTrends = sCommonTrends(:,:,ceil(Quant*M));
qCycles       = sCycles(:,:,ceil(Quant*M));
qTrends       = sTrends(:,:,ceil(Quant*M));
qTrendsReal   = sTrendsReal(:,:,ceil(Quant*M));

% Convenience series (Model2 decomposition)
Pi_bar = squeeze(CommonTrends(:,1,:));
R_bar  = squeeze(CommonTrends(:,2,:) - CommonTrends(:,4,:)); % r* - cy*
M_bar  = squeeze(CommonTrends(:,2,:));                       % m* = r* + cy*
Cy_bar = squeeze(CommonTrends(:,4,:));
Ts_bar = squeeze(CommonTrends(:,3,:));

sPi_bar = sort(Pi_bar,2);
sR_bar  = sort(R_bar,2);
sM_bar  = sort(M_bar,2);
sCy_bar = sort(Cy_bar,2);
sTs_bar = sort(Ts_bar,2);

qPi_bar = sPi_bar(:,ceil(Quant*M));
qR_bar  = sR_bar(:,ceil(Quant*M));
qM_bar  = sM_bar(:,ceil(Quant*M));
qCy_bar = sCy_bar(:,ceil(Quant*M));
qTs_bar = sTs_bar(:,ceil(Quant*M));

% --- Write CSV and MAT for charts ---
Ytr = [Y(:,2), Y(:,3)-Y(:,2)+Y(:,6)-Y(:,5), Y(:,5)-Y(:,3), Y(:,6)-Y(:,5)];
save(fullfile(FigSubFolder,'OutMod2forCharts.mat'),'Time','qR_bar','qCy_bar','qM_bar','qPi_bar','qTs_bar','y','-v7.3');

Tdatetime = datetime(Time,'ConvertFrom','datenum');
tbl = table(Tdatetime, ...
    qPi_bar(:,3), qPi_bar(:,1), qPi_bar(:,2), qPi_bar(:,4), qPi_bar(:,5), ...
    qR_bar(:,3),  qR_bar(:,1),  qR_bar(:,2),  qR_bar(:,4),  qR_bar(:,5), ...
    qM_bar(:,3),  qM_bar(:,1),  qM_bar(:,2),  qM_bar(:,4),  qM_bar(:,5), ...
    qCy_bar(:,3), qCy_bar(:,1), qCy_bar(:,2), qCy_bar(:,4), qCy_bar(:,5), ...
    qTs_bar(:,3), qTs_bar(:,1), qTs_bar(:,2), qTs_bar(:,4), qTs_bar(:,5), ...
    'VariableNames', {'Date', ...
    'Pi_bar_med','Pi_bar_p2_5','Pi_bar_p16','Pi_bar_p84','Pi_bar_p97_5', ...
    'R_bar_med','R_bar_p2_5','R_bar_p16','R_bar_p84','R_bar_p97_5', ...
    'M_bar_med','M_bar_p2_5','M_bar_p16','M_bar_p84','M_bar_p97_5', ...
    'Cy_bar_med','Cy_bar_p2_5','Cy_bar_p16','Cy_bar_p84','Cy_bar_p97_5', ...
    'Ts_bar_med','Ts_bar_p2_5','Ts_bar_p16','Ts_bar_p84','Ts_bar_p97_5'});
writetable(tbl, fullfile(FigSubFolder,'OutMod2forCharts.csv'));

% --- Figures (PDFs), mirroring original Model2 plots ---------------------
f = figure; subplot(1,3,1); PlotStatesShaded(Time,qR_bar); axis([-inf inf -.5 4.5]); title('R=M-Cy');
subplot(1,3,2); PlotStatesShaded(Time,qM_bar); axis([-inf inf -.5 4.5]); title('M');
subplot(1,3,3); PlotStatesShaded(Time,qCy_bar); axis([-inf inf -.5 4.5]); title('Cy');
printpdf(gcf, fullfile(FigSubFolder,'Rdecomp.pdf')); close(gcf);

f = figure; PlotStatesShaded(Time,qPi_bar); title('\pi^*'); printpdf(f, fullfile(FigSubFolder,'PIbar.pdf'));
f = figure; PlotStatesShaded(Time,qPi_bar); hold on;
plot(Time,y(:,2),'b-','linewidth',2.0); plot(Time,y(:,1),'b:','linewidth',1);
title('\pi^* and \pi'); hold off; printpdf(f, fullfile(FigSubFolder,'PIbar_obs.pdf'));

f = figure; PlotStatesShaded(Time,qM_bar); title('m^*'); printpdf(f, fullfile(FigSubFolder,'Mbar.pdf'));
f = figure; PlotStatesShaded(Time,qM_bar); hold on;
plot(Time,y(:,3)-y(:,2)+y(:,6)-y(:,5),'b:','linewidth',1); hold off;
title('m^*, r-\pi^e + (baa-r^L)'); printpdf(f, fullfile(FigSubFolder,'Mbar_obs.pdf'));

f = figure; PlotStatesShaded(Time,qR_bar); title('r^*'); printpdf(f, fullfile(FigSubFolder,'Rbar.pdf'));
f = figure; PlotStatesShaded(Time,qR_bar); hold on;
plot(Time,y(:,4)-y(:,2),'b*-','linewidth',2.0);
plot(Time,y(:,3)-y(:,2),'b:','linewidth',1); hold off;
title('m^*-cy^*, r-\pi^e and r^e-\pi^e'); printpdf(f, fullfile(FigSubFolder,'Rbar_obs.pdf'));

f = figure; PlotStatesShaded(Time,qTs_bar); title('Ts^*'); printpdf(f, fullfile(FigSubFolder,'TSbar.pdf'));
f = figure; PlotStatesShaded(Time,qTs_bar); hold on;
plot(Time,y(:,5)-y(:,3),'b:','linewidth',1); hold off;
title('ts^*, r^L-r'); printpdf(f, fullfile(FigSubFolder,'TSbar_obs.pdf'));

f = figure; PlotStatesShaded(Time,-qCy_bar); axis([-inf inf -3 1]); printpdf(f, fullfile(FigSubFolder,'CYscaled.pdf'));
f = figure; PlotStatesShaded(Time,qM_bar);  axis([-inf inf 1 5]);  printpdf(f, fullfile(FigSubFolder,'Mscaled.pdf'));
f = figure; PlotStatesShaded(Time,qR_bar);  axis([-inf inf -.5 3.5]); printpdf(f, fullfile(FigSubFolder,'Rscaled.pdf'));
f = figure; PlotStatesShaded(Time,qTs_bar); axis([-inf inf -.5 3.5]); printpdf(f, fullfile(FigSubFolder,'TSscaled.pdf'));

disp('Post-processing complete: wrote CSV/MAT and PDFs in FiguresModel2/')

% ===================== Local Function ===================================
function run_one_chain_M2(chain_id, seed, Ndraws, THIN, S, out_file)
    rng(seed,'combRecursive')

    y=S.y; C=S.C; A=S.A; R=S.R; Q=S.Q; S0=S.S0; P0=S.P0;
    r=S.r; n=S.n; p=S.p; rn=S.rn; b0=S.b0; df0tr=S.df0tr;
    SC0tr=S.SC0tr; Psi=S.Psi; Y=S.Y; Time=S.Time; Mnem=S.Mnem;
    S0tr = S.S0tr; P0tr = S.P0tr; SAVE_LEVEL = S.SAVE_LEVEL;

    Nkeep = floor(Ndraws/THIN);
    States     = nan(size(y,1), rn, Nkeep);
    Trends     = nan(size(y,1), n , Nkeep);
    TrendsReal = nan(size(y,1), n , Nkeep);
    if strcmpi(SAVE_LEVEL,'full')
        LogLik = nan(1, Nkeep); SS0 = nan(r, Nkeep);
        AA = nan(rn, rn, Nkeep); QQ = nan(rn, rn, Nkeep);
        CC = nan(n , rn, Nkeep); RR = nan(n , n , Nkeep);
        P_acc = nan(1, Ndraws);
    end

    notrend = [];   % as in original Model2 (no MH step on nontrends)
    keep_idx = 0;
    t0 = tic;

    for jm = 1:Ndraws
        kf = KF(y,C,R,A,Q,S0,P0);
        loglik = kf.LogLik;

        % (optional) random-walk S0 on zero-variance trends — disabled by default

        kc = KC(kf);
        % BVAR step for cycle
        Ycyc = kc.S(:,r+1:r+n);
        for jp=1:p
            Ycyc = [kc.S0(r+(jp-1)*n+1:r+n*jp)'; Ycyc];
        end
        [beta,sigma] = BVAR(Ycyc,p,b0,Psi,.2,1);
        A(r+1:r+n,r+1:end) = beta';
        Q(r+1:r+n,r+1:r+n) = sigma;

        % Trend covariances
        Ytr  = [kc.S0(1:r)'; kc.S(:,1:r)];
        SCtr = CovarianceDraw(diff(Ytr), df0tr, diag(SC0tr));
        Q(1:r,1:r) = SCtr;

        % steady-state P0 for cycle (robust)
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

        % thinning
        if mod(jm,THIN)==0
            keep_idx = keep_idx + 1;
            States(:,:,keep_idx)     = kc.S;
            Trends(:,:,keep_idx)     = kc.S(:,1:r)*C(:,1:r)';
            TrendsReal(:,:,keep_idx) = kc.S(:,2:r)*C(:,2:r)';
            if strcmpi(SAVE_LEVEL,'full')
                LogLik(keep_idx) = loglik;
                SS0(:,keep_idx)  = S0(1:r);
                AA(:,:,keep_idx) = A;
                QQ(:,:,keep_idx) = Q;
                CC(:,:,keep_idx) = C;
                RR(:,:,keep_idx) = R;
            end
        end

        if mod(jm,1000)==0
            fprintf('[chain %d] %d/%d, elapsed %.1f sec, kept %d\n', ...
                chain_id, jm, Ndraws, toc(t0), keep_idx);
        end
    end

    % trim if partial
    if keep_idx < Nkeep
        States(:,:,keep_idx+1:end) = [];
        Trends(:,:,keep_idx+1:end) = [];
        TrendsReal(:,:,keep_idx+1:end) = [];
        if strcmpi(SAVE_LEVEL,'full')
            LogLik(:,keep_idx+1:end) = [];
            SS0(:,keep_idx+1:end) = [];
            AA(:,:,keep_idx+1:end) = [];
            QQ(:,:,keep_idx+1:end) = [];
            CC(:,:,keep_idx+1:end) = [];
            RR(:,:,keep_idx+1:end) = [];
        end
    end

    CommonTrends = States(:,1:r,:);
    Cycles       = States(:,r+1:r+n,:);

    % per-chain save
    if strcmpi(SAVE_LEVEL,'lite')
        save(out_file,'CommonTrends','Trends','TrendsReal','Cycles', ...
            'SC0tr','S0tr','P0tr','df0tr','Psi','Time','Y','y','Mnem','THIN','Ndraws','-v7.3');
    else
        save(out_file,'CommonTrends','Trends','TrendsReal','Cycles', ...
            'AA','QQ','CC','RR','LogLik','SS0','P_acc', ...
            'SC0tr','S0tr','P0tr','df0tr','Psi','Time','Y','y','Mnem','THIN','Ndraws','-v7.3');
    end
end
