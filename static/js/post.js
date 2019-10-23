$(function() {

  function buildHTML(post){
    // var html = ''
    // if (post.same == 1) {
    //   var html_same = `<div class="post-content card col-sm-11 col-md-11 bg-info" data-post-id=${post.id}>`
    // } else { 
    //   var html_same = `<div class="post-content card col-sm-11 col-md-11" data-post-id=${post.id}>`
    // }
    // var html = `<div class="card-body">
    //   <h5 class="card-title">${post.title}</h5>
    //   <h6 class="card-subtitle mb-2">投稿日: ${post.date_posted} By ${post.authorname}</h6>
    //   <p class="card-text">${post.content}</p>
    //   </div>
    //   </div>`
    // html = html_same + html
    if (post.same == 1){
      var html = `<div class="media post-right" data-post-id=${post.id}>
        <div class="balloon1-right">
          <h5>${post.title}  <small>by ${post.authorname}</small></h5>
          <p>${post.content}</p>
        </div>
      </div>`
    } else if (post.authorname == 'HarryBotter') {
      var html = `<div class="media" data-post-id=${post.id}>
        <img src="static/images/chatbot_icon.png" class="mr-3 post-pict" alt="...">
        <div class="balloon1-left">
            <h5>${post.title}  <small>by ${post.authorname}</small></h5>
            <p>${post.content}</p>
        </div>
      </div>`
    } else {
      var html = `<div class="media" data-post-id=${post.id}>
        <img src="static/images/user_icon.png" class="mr-3 post-pict" alt="...">
        <div class="balloon1-left">
          <h5>${post.title}  <small>by ${post.authorname}</small></h5>
          <p>${post.content}</p>
        </div>
      </div>`
    }
    return html
  }

  $('#submit').on('click', function(e){
    e.preventDefault();
    console.log('hoge');
    $.ajax({
      type: 'POST',
      url: '/post_ajax',
      data: $(this).parent().parent('form').serialize(),
    })
    .done(function (data) {
      console.log('Ajax成功!')
      var html = buildHTML(data)
      $('#message-box').append(html)
      $('#post-form').get(0).reset();
      $('html, body').animate({scrollTop:$('#message-box')[0].scrollHeight});
    })
    .fail(function() {
      alert('Ajaxに失敗しました');
    })
  })
  var reloadMessages = function() {
    // lastメソッドはidではなくクラス指定(class="~" => $('.~:last'))出ないと効かない!
    // last_post_id = $('#post-content').last().data('post-id')
    var last_post_id = $('.media:last').data('post-id')
    console.log(last_post_id)
    $.ajax({
      //ルーティングで設定した通り/groups/id番号/api/messagesとなるよう文字列を書く
      url: '/post_api',
      //ルーティングで設定した通りhttpメソッドをgetに指定
      type: 'POST',
      // dataType: 'json',
      // //dataオプションでリクエストに値を含める
      // data: {id: last_post_id}
      data: JSON.stringify({"id":last_post_id}),
      contentType: 'application/json',
    })
    .done(function(posts) {
      //追加するHTMLの入れ物を作る
      var insertHTML = '';

      //メッセージが入ったHTMLを取得
      var messege_box = $('#message-box')

      //配列messagesの中身一つ一つを取り出し、HTMLに変換したものを入れ物に足し合わせる
      console.log(posts)

      if ($.isEmptyObject(posts) == false) {
        posts.forEach(function(post){
          var html = buildHTML(post)
          $(messege_box).append(html);
        })
        $('html, body').animate({scrollTop:$('#message-box')[0].scrollHeight});
        console.log('animated')
      } 
    })
    .fail(function() {
      alert('error');
    });
  };
  setInterval(reloadMessages, 10000);
})